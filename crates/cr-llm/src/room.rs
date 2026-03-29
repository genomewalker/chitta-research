/// Multi-model discussion room — no external dependencies.
///
/// A `DiscussionRoom` holds N participants (each an `Arc<dyn LlmClient>` with
/// a name and system prompt). When `complete()` is called, the room runs
/// `rounds` of async debate: in each round every participant reads the full
/// thread so far and posts a response in parallel. After all rounds a
/// dedicated `Synthesizer` participant distills the thread into a final answer.
///
/// This implements `LlmClient` so it is a drop-in replacement anywhere a
/// single model is expected.
use std::sync::Arc;
use async_trait::async_trait;
use tracing::debug;

use crate::{CompletionRequest, CompletionResponse, LlmClient, LlmError, Message, TokenUsage};

/// One participant in a discussion room.
pub struct Participant {
    pub name: String,
    /// System prompt that gives the participant its distinct perspective.
    /// Good prompts: "You are a pessimist. Find every way this hypothesis
    /// could be wrong.", "You are an empiricist. Focus on testability."
    pub system: String,
    pub client: Arc<dyn LlmClient>,
}

/// A multi-model discussion room that implements `LlmClient`.
///
/// Usage:
/// ```rust,no_run
/// use std::sync::Arc;
/// use cr_llm::{room::DiscussionRoom, room::Participant, ClaudeCliClient};
///
/// let room = DiscussionRoom::builder("Evaluate this hypothesis")
///     .add("Critic",    "Find every flaw in the hypothesis.",
///          Arc::new(ClaudeCliClient::new()))
///     .add("Empiricist","Focus on testability and measurement.",
///          Arc::new(ClaudeCliClient::with_model("claude-sonnet-4-6".to_string())))
///     .rounds(2)
///     .build();
/// ```
pub struct DiscussionRoom {
    pub topic: String,
    pub participants: Vec<Participant>,
    /// Number of debate rounds before synthesis. Default: 2.
    pub rounds: usize,
    /// Synthesizer: a dedicated participant that reads the full thread and
    /// produces the final structured output expected by the caller.
    /// If None, the last message in the thread is returned verbatim.
    pub synthesizer: Option<Participant>,
}

impl DiscussionRoom {
    pub fn builder(topic: impl Into<String>) -> DiscussionRoomBuilder {
        DiscussionRoomBuilder {
            topic: topic.into(),
            participants: Vec::new(),
            rounds: 2,
            synthesizer: None,
        }
    }
}

pub struct DiscussionRoomBuilder {
    topic: String,
    participants: Vec<Participant>,
    rounds: usize,
    synthesizer: Option<Participant>,
}

impl DiscussionRoomBuilder {
    pub fn add(
        mut self,
        name: impl Into<String>,
        system: impl Into<String>,
        client: Arc<dyn LlmClient>,
    ) -> Self {
        self.participants.push(Participant {
            name: name.into(),
            system: system.into(),
            client,
        });
        self
    }

    pub fn rounds(mut self, n: usize) -> Self {
        self.rounds = n;
        self
    }

    pub fn synthesizer(
        mut self,
        system: impl Into<String>,
        client: Arc<dyn LlmClient>,
    ) -> Self {
        self.synthesizer = Some(Participant {
            name: "Synthesizer".into(),
            system: system.into(),
            client,
        });
        self
    }

    pub fn build(self) -> DiscussionRoom {
        DiscussionRoom {
            topic: self.topic,
            participants: self.participants,
            rounds: self.rounds,
            synthesizer: self.synthesizer,
        }
    }
}

/// Mock room: uses MockLlmClient for all participants.
/// Exercises the full debate loop (thread building, parallel rounds, synthesis)
/// without any external calls. Pass `--mock` to cr-daemon to activate.
pub fn mock_room(topic: impl Into<String>) -> DiscussionRoomBuilder {
    use crate::MockLlmClient;
    DiscussionRoom::builder(topic)
        .add("Critic",     "Find every flaw in the hypothesis.",   Arc::new(MockLlmClient::new()))
        .add("Empiricist", "Focus on testability.",                Arc::new(MockLlmClient::new()))
        .rounds(2)
        .synthesizer(
            "Synthesize the debate into final structured JSON output.",
            Arc::new(MockLlmClient::new()),
        )
}

/// Preset: build a room with the standard claude-cli / codex participants.
/// Each gets a distinct analytical role so the debate is genuinely divergent.
pub fn standard_room(topic: impl Into<String>) -> DiscussionRoomBuilder {
    let topic = topic.into();
    DiscussionRoom::builder(topic)
        .add(
            "Critic",
            "You are a scientific critic. Find every flaw, untested assumption, \
             and confound in the hypothesis or plan. Be specific and brief.",
            Arc::new(crate::ClaudeCliClient::new()),
        )
        .add(
            "Empiricist",
            "You are an empiricist. Focus on testability: is the experiment \
             actually measurable, reproducible, and falsifiable? Propose concrete \
             improvements to the experimental design.",
            Arc::new(crate::CodexClient::new()),
        )
        .rounds(2)
        .synthesizer(
            "You are a JSON synthesizer. Read the debate above, resolve any contradictions between participants, and produce the final \
             answer the original prompt requested. Incorporate valid criticisms. \
             CRITICAL: Respond with ONLY raw JSON — no prose, no explanation, no \
             [brackets], no markdown fences. Start your response with the first \
             character of the JSON literal (either { or [).",
            Arc::new(crate::ClaudeCliClient::new()),
        )
}

/// Format the full discussion thread for a participant to read.
/// Each post is prefixed with the poster's name.
fn build_context(topic: &str, thread: &[(String, String)], viewer: &str) -> String {
    let mut ctx = format!("Topic: {topic}\n\n");
    for (name, content) in thread {
        if name == viewer {
            ctx.push_str(&format!("You previously said:\n{content}\n\n"));
        } else {
            ctx.push_str(&format!("{name}:\n{content}\n\n"));
        }
    }
    ctx
}

#[async_trait]
impl LlmClient for DiscussionRoom {
    async fn complete(&self, req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        // The user message is the actual prompt passed by the agent.
        // We use it as the opening post in the thread.
        let opening = req.messages.iter()
            .filter(|m| m.role == "user")
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut thread: Vec<(String, String)> = vec![
            ("PROMPT".into(), opening),
        ];

        let mut total_input = 0u64;
        let mut total_output = 0u64;

        // Debate rounds — all participants respond in parallel each round.
        for round in 0..self.rounds {
            debug!(round, participants = self.participants.len(), "room: starting round");

            let mut handles = Vec::new();
            for participant in &self.participants {
                let context = build_context(&self.topic, &thread, &participant.name);
                let r = CompletionRequest {
                    model: req.model.clone(),
                    system: participant.system.clone(),
                    messages: vec![Message { role: "user".into(), content: context }],
                    max_tokens: req.max_tokens,
                    temperature: req.temperature,
                };
                let client = Arc::clone(&participant.client);
                handles.push((participant.name.clone(), tokio::spawn(async move {
                    client.complete(r).await
                })));
            }

            for (name, handle) in handles {
                match handle.await {
                    Ok(Ok(resp)) => {
                        debug!(participant = %name, round, "room: got response");
                        total_input += resp.usage.input;
                        total_output += resp.usage.output;
                        thread.push((name, resp.content));
                    }
                    Ok(Err(e)) => {
                        // Non-fatal: log and continue with other participants
                        debug!(participant = %name, error = %e, "room: participant error");
                        continue;
                    }
                    Err(e) => {
                        debug!(participant = %name, error = %e, "room: task panicked");
                        continue;
                    }
                }
            }
        }

        // Synthesis: produce the final structured output.
        let synthesis = if let Some(synth) = &self.synthesizer {
            let context = build_context(&self.topic, &thread, "Synthesizer");
            let r = CompletionRequest {
                model: req.model.clone(),
                system: synth.system.clone(),
                messages: vec![Message { role: "user".into(), content: context }],
                max_tokens: req.max_tokens,
                temperature: 0.2, // low temp for reliable structured output
            };
            let resp = synth.client.complete(r).await?;
            total_input  += resp.usage.input;
            total_output += resp.usage.output;
            resp.content
        } else {
            // No synthesizer: return the last participant's response
            thread.last()
                .map(|(_, c)| c.clone())
                .unwrap_or_default()
        };

        Ok(CompletionResponse {
            content: synthesis,
            usage: TokenUsage { input: total_input, output: total_output },
            // Expose the full debate thread so callers can persist it
            debate_thread: thread,
        })
    }
}
