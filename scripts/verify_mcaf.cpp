// verify_mcaf.cpp — MCAF compression + speed verifier for chitta-research.
//
// Emits a VerificationResult JSON to stdout (one object).
// No htslib required. Links: -lzstd
//
// Build:
//   g++ -O2 -std=c++17 -I/maps/projects/fernandezguerra/apps/repos/mcaf \
//       verify_mcaf.cpp -lzstd -o verify_mcaf
//
// Usage:
//   verify_mcaf [--mcaf PATH] [--bam PATH] [--mode ratio|speed] [--threads N]

#include "mcaf.hpp"
#include <zstd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

static const char* DEFAULT_MCAF = "/maps/projects/caeg/scratch/kbd606/bam-filter/viking/LV7008867351.sorted.mcaf";
static const char* DEFAULT_BAM  = "/maps/projects/caeg/scratch/kbd606/bam-filter/viking/LV7008867351.sorted.bam";

// ── helpers ───────────────────────────────────────────────────────────────────

static int64_t file_size(const char* path) {
    struct stat st{};
    if (stat(path, &st) != 0) return -1;
    return (int64_t)st.st_size;
}

static void emit_invalid(const char* note) {
    printf("{\n"
           "  \"status\": {\"kind\": \"invalid\"},\n"
           "  \"metrics\": {},\n"
           "  \"notes\": \"%s\"\n"
           "}\n", note);
}

static void emit_json(const char* status_kind,
                      double metric_val, const char* metric_name,
                      double extra1_val, const char* extra1_name,
                      double extra2_val, const char* extra2_name,
                      const char* notes)
{
    printf("{\n"
           "  \"status\": {\"kind\": \"%s\"},\n"
           "  \"metrics\": {\n"
           "    \"%s\": %.4f,\n"
           "    \"%s\": %.2f,\n"
           "    \"%s\": %.2f\n"
           "  },\n"
           "  \"notes\": \"%s\"\n"
           "}\n",
           status_kind,
           metric_name, metric_val,
           extra1_name, extra1_val,
           extra2_name, extra2_val,
           notes);
}

// ── ratio mode ────────────────────────────────────────────────────────────────

static void mode_ratio(const char* mcaf_path, const char* bam_path) {
    int64_t mcaf_bytes = file_size(mcaf_path);
    int64_t bam_bytes  = file_size(bam_path);

    if (bam_bytes < 0) {
        char msg[512];
        snprintf(msg, sizeof(msg), "BAM not found: %s", bam_path);
        emit_invalid(msg);
        return;
    }
    if (mcaf_bytes < 0) {
        char msg[512];
        snprintf(msg, sizeof(msg), "MCAF not found: %s", mcaf_path);
        emit_invalid(msg);
        return;
    }

    double mcaf_mb  = (double)mcaf_bytes / (1024.0 * 1024.0);
    double bam_mb   = (double)bam_bytes  / (1024.0 * 1024.0);
    double ratio    = bam_mb / mcaf_mb;   // >1 = MCAF smaller than BAM
    double saved_mb = bam_mb - mcaf_mb;

    const char* status = (ratio > 1.0) ? "pass" : "fail";
    char notes[256];
    snprintf(notes, sizeof(notes),
             "mcaf=%.1f MB  bam=%.1f MB  ratio=%.4f  saved=%.1f MB",
             mcaf_mb, bam_mb, ratio, saved_mb);

    emit_json(status,
              ratio,    "compression_ratio",
              mcaf_mb,  "mcaf_size_mb",
              bam_mb,   "bam_size_mb",
              notes);
}

// ── speed mode ────────────────────────────────────────────────────────────────
// Reads and decompresses every ZSTD frame in the MCAF to measure decode throughput.

static void mode_speed(const char* mcaf_path) {
    int fd = open(mcaf_path, O_RDONLY);
    if (fd < 0) {
        char msg[512];
        snprintf(msg, sizeof(msg), "cannot open: %s", mcaf_path);
        emit_invalid(msg);
        return;
    }

    // Read entire file into memory for pure decode benchmark (no I/O jitter).
    int64_t fsize = file_size(mcaf_path);
    if (fsize <= 0) { close(fd); emit_invalid("empty or missing MCAF"); return; }

    std::vector<uint8_t> buf((size_t)fsize);
    {
        ssize_t total = 0;
        while (total < fsize) {
            ssize_t n = read(fd, buf.data() + total, (size_t)(fsize - total));
            if (n <= 0) break;
            total += n;
        }
    }
    close(fd);

    // Walk ZSTD frames and decompress each.
    size_t out_buf_cap = 64 * 1024 * 1024; // 64 MB decompression buffer
    std::vector<uint8_t> out_buf(out_buf_cap);

    uint64_t frames_decoded = 0;
    uint64_t bytes_decompressed = 0;

    auto t0 = std::chrono::steady_clock::now();

    const uint8_t* p   = buf.data();
    const uint8_t* end = buf.data() + buf.size();

    while (p < end) {
        // Find next ZSTD magic (0xFD2FB528 little-endian)
        if ((size_t)(end - p) < 4) break;
        uint32_t magic;
        memcpy(&magic, p, 4);
        if (magic != 0xFD2FB528u) {
            // Skip one byte to find next frame
            ++p;
            continue;
        }

        // Get frame content size
        size_t frame_size = ZSTD_findFrameCompressedSize(p, (size_t)(end - p));
        if (ZSTD_isError(frame_size)) { ++p; continue; }

        // Grow decompression buffer if needed
        size_t dec_bound = ZSTD_getFrameContentSize(p, frame_size);
        if (dec_bound == ZSTD_CONTENTSIZE_UNKNOWN || dec_bound == ZSTD_CONTENTSIZE_ERROR) {
            dec_bound = out_buf_cap; // fallback
        }
        if (dec_bound > out_buf.size()) {
            out_buf.resize(dec_bound + (4 * 1024 * 1024));
        }

        size_t n = ZSTD_decompress(out_buf.data(), out_buf.size(), p, frame_size);
        if (!ZSTD_isError(n)) {
            bytes_decompressed += n;
            ++frames_decoded;
        }
        p += frame_size;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double throughput  = (bytes_decompressed / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    char notes[256];
    snprintf(notes, sizeof(notes),
             "frames=%llu  decompressed=%.1f MB  elapsed=%.0f ms  throughput=%.0f MB/s",
             (unsigned long long)frames_decoded,
             bytes_decompressed / (1024.0 * 1024.0),
             elapsed_ms,
             throughput);

    emit_json("pass",
              elapsed_ms,  "scan_time_ms",
              throughput,  "throughput_mb_s",
              (double)frames_decoded, "frames_decoded",
              notes);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* mcaf_path = DEFAULT_MCAF;
    const char* bam_path  = DEFAULT_BAM;
    const char* mode      = "ratio";

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--mcaf=", 7) == 0)  mcaf_path = argv[i] + 7;
        else if (strncmp(argv[i], "--bam=",  6) == 0)  bam_path  = argv[i] + 6;
        else if (strncmp(argv[i], "--mode=", 7) == 0)  mode      = argv[i] + 7;
        else if (strcmp(argv[i], "--mcaf") == 0 && i+1 < argc) mcaf_path = argv[++i];
        else if (strcmp(argv[i], "--bam")  == 0 && i+1 < argc) bam_path  = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i+1 < argc) mode      = argv[++i];
    }

    if (strcmp(mode, "speed") == 0)
        mode_speed(mcaf_path);
    else
        mode_ratio(mcaf_path, bam_path);

    return 0;
}
