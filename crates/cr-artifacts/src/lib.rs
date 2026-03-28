use git2::{IndexAddOption, Repository, Signature};
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("git error: {0}")]
    Git(#[from] git2::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("artifact not found: {commit}:{path}")]
    NotFound { commit: String, path: String },
    #[error("lock poisoned")]
    LockPoisoned,
}

pub struct ArtifactStore {
    repo: Mutex<Repository>,
}

impl ArtifactStore {
    pub fn open_or_init(path: &Path) -> Result<Self, ArtifactError> {
        let repo = if path.join(".git").exists() {
            Repository::open(path)?
        } else {
            std::fs::create_dir_all(path)?;
            let repo = Repository::init(path)?;
            {
                let sig = Signature::now("chitta-research", "chitta@local")?;
                let tree_id = repo.index()?.write_tree()?;
                let tree = repo.find_tree(tree_id)?;
                repo.commit(Some("HEAD"), &sig, &sig, "init artifact store", &tree, &[])?;
            }
            repo
        };
        Ok(Self { repo: Mutex::new(repo) })
    }

    fn lock_repo(&self) -> Result<std::sync::MutexGuard<'_, Repository>, ArtifactError> {
        self.repo.lock().map_err(|_| ArtifactError::LockPoisoned)
    }

    pub fn commit_run_artifacts(
        &self,
        run_id: &str,
        files: &[(&str, &[u8])],
        message: &str,
    ) -> Result<String, ArtifactError> {
        let repo = self.lock_repo()?;
        let workdir = repo.workdir().expect("bare repos not supported");

        let run_dir = workdir.join(run_id);
        std::fs::create_dir_all(&run_dir)?;

        for (name, data) in files {
            let file_path = run_dir.join(name);
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file_path, data)?;
        }

        let mut index = repo.index()?;
        index.add_all(["."], IndexAddOption::DEFAULT, None)?;
        index.write()?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;

        let sig = Signature::now("chitta-research", "chitta@local")?;
        let head = repo.head()?.peel_to_commit()?;
        let oid = repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &[&head])?;

        Ok(oid.to_string())
    }

    pub fn tag_run(&self, run_id: &str, oid: &str) -> Result<(), ArtifactError> {
        let repo = self.lock_repo()?;
        let obj = repo.revparse_single(oid)?;
        let sig = Signature::now("chitta-research", "chitta@local")?;
        repo.tag(&format!("run/{run_id}"), &obj, &sig, run_id, false)?;
        Ok(())
    }

    pub fn read_artifact(&self, commit: &str, path: &str) -> Result<Vec<u8>, ArtifactError> {
        let repo = self.lock_repo()?;
        let obj = repo.revparse_single(commit)?;
        let commit_obj = obj.peel_to_commit()?;
        let tree = commit_obj.tree()?;
        let entry = tree.get_path(Path::new(path)).map_err(|_| ArtifactError::NotFound {
            commit: commit.to_string(),
            path: path.to_string(),
        })?;
        let blob = repo.find_blob(entry.id())?;
        Ok(blob.content().to_vec())
    }
}
