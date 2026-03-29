PREFIX ?= $(HOME)/.local
BINDIR  = $(PREFIX)/bin
REPO    = $(shell pwd)

# Build target dir on local disk to avoid NFS rlib truncation
export CARGO_TARGET_DIR ?= /tmp/cr-target
export PATH := /usr/bin:$(HOME)/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$(PATH)

.PHONY: build install uninstall clean release

## build — compile all binaries in dev mode
build:
	unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER; \
	cargo build --bins

## release — compile all binaries in release mode
release:
	unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER; \
	cargo build --bins --release

## install — build release binaries and install to PREFIX/bin (default: ~/.local/bin)
##   cresearch   — the research daemon
##   cr-report   — HTML report generator
install: release
	mkdir -p $(BINDIR)
	install -m755 $(CARGO_TARGET_DIR)/release/cresearch  $(BINDIR)/cresearch
	install -m755 $(CARGO_TARGET_DIR)/release/cr-report   $(BINDIR)/cr-report
	@echo "Installed to $(BINDIR):"
	@echo "  cresearch --agenda agenda.yaml [--max-cycles N]"
	@echo "  cr-report  --graph graph_state.json"

## uninstall — remove installed binaries
uninstall:
	rm -f $(BINDIR)/cresearch $(BINDIR)/cr-report

## clean — remove local build artifacts
clean:
	rm -rf /tmp/cr-target
