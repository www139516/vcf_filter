#!/usr/bin/env python3
# coding:utf-8
import argparse
import os
import sys
import platform
import logging
import time
import gzip
import csv
import re
import shutil
import itertools
from datetime import timedelta
from functools import wraps
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Any

# import pysam for tabix
try:
    import pysam

    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

# --- Constants ---
SCRIPT_VERSION = "1.6.0"
TABIX_INDEX_SUFFIX = ".tbi"
BLOCK_SIZE = 1048576  # 1MB block size

# Logging is initialized in main() via setup_logging()
# (console + a timestamped log file per run)
logger = logging.getLogger("vcf_processor")
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(log_dir: str = "") -> str:
    """Initialize logging: output to both console and file.

    Each run creates an independent timestamped log file
    (vcf_processor_YYYYMMDD_HHMMSS.log) so that a specific run can be
    located easily during later debugging. Returns the log file path.
    """
    log_dir = log_dir or os.getcwd()
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"vcf_processor_{timestamp}.log")

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return log_path


def log_run_banner(log_path: str, args) -> None:
    """Record environment and command information at the top of the log,
    so that problems can be reproduced during later debugging."""
    logger.info("=" * 70)
    logger.info(f"VCF Processor v{SCRIPT_VERSION}")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    logger.info(f"Python: {platform.python_version()} ({sys.executable})")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(
        f"pysam: {pysam.__version__ if PYSAM_AVAILABLE else 'not installed (optional dependency)'}"
    )
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Input file: {os.path.abspath(args.vcf)}")
    logger.info("=" * 70)


def install_excepthook() -> None:
    """Write uncaught exceptions to the log in full before exiting."""

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _hook

# --- Helper Functions ---


def timer_decorator(func):
    """Decorater, record the time for excuting each process"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = timedelta(seconds=end_time - start_time)
        logger.info(f"{func.__name__} finished, time: {elapsed}")
        return result

    return wrapper


def validate_file_exists(file_path: str, file_type: str = "input file") -> bool:
    """Verify that a file exists and is readable"""
    if not os.path.exists(file_path):
        logger.error(f"{file_type} does not exist: {file_path}")
        sys.exit(1)
    if not os.path.isfile(file_path):
        logger.error(f"{file_type} is not a file: {file_path}")
        sys.exit(1)
    if not os.access(file_path, os.R_OK):
        logger.error(f"No read permission for {file_type}: {file_path}")
        sys.exit(1)
    return True


def validate_directory(directory: str, create_if_missing: bool = True) -> bool:
    """Verify that a directory exists; create it if missing"""
    if not os.path.exists(directory):
        if create_if_missing:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created output directory: {directory}")
                return True
            except OSError as e:
                logger.error(f"Cannot create directory {directory}: {str(e)}")
                sys.exit(1)
        else:
            logger.error(f"Directory does not exist: {directory}")
            sys.exit(1)
    if not os.path.isdir(directory):
        logger.error(f"{directory} is not a directory")
        sys.exit(1)
    if not os.access(directory, os.W_OK):
        logger.error(f"No write permission for directory {directory}")
        sys.exit(1)
    return True


def check_pysam_available(required_functionality: str):
    """Check whether pysam is available; give a friendly hint if not"""
    if not PYSAM_AVAILABLE:
        logger.error(
            f"The pysam library is required for {required_functionality}.\n"
            "Please install it with: pip install pysam"
        )
        sys.exit(1)


def open_file_smart(file_path: str, mode: str = "r"):
    """Open a file smartly, with support for gzip-compressed files"""
    try:
        if file_path.endswith(".gz"):
            # For write mode, use compression level 6 (balance of speed and ratio)
            if "w" in mode:
                return gzip.open(file_path, mode + "t", compresslevel=6)
            else:
                return gzip.open(file_path, mode + "t")
        else:
            return open(file_path, mode)
    except IOError as e:
        logger.error(f"Cannot open file {file_path}: {str(e)}")
        sys.exit(1)


def get_index_path(vcf_path: str) -> str:
    """Get the index file path"""
    if vcf_path.endswith(".gz"):
        return vcf_path + TABIX_INDEX_SUFFIX
    return vcf_path + ".idx"


def create_tabix_index(vcf_path: str, force: bool = False, fatal: bool = True) -> bool:
    """Create a tabix index with pysam.

    fatal=True (command line --create-index): exit on missing dependency
    or failure.
    fatal=False (automatic index creation inside the processing pipeline):
    degrade to a warning without affecting results already produced.
    """
    if not PYSAM_AVAILABLE:
        msg = (
            "The pysam library is required for the tabix indexing feature; "
            "index creation skipped.\n"
            "Please install it with: pip install pysam"
        )
        (logger.error if fatal else logger.warning)(msg)
        if fatal:
            sys.exit(1)
        return False

    if not vcf_path.endswith(".gz"):
        logger.error("Tabix indexes can only be created for gzip-compressed VCF files (.vcf.gz)")
        return False

    index_path = get_index_path(vcf_path)

    if os.path.exists(index_path) and not force:
        logger.info(f"Index file already exists: {index_path}; use --force to rebuild")
        return True

    try:
        pysam.tabix_index(
            vcf_path, force=force, preset="vcf", index_filename=index_path
        )
        if os.path.exists(index_path):
            logger.info(f"Successfully created tabix index: {index_path}")
            return True
        else:
            logger.error("Failed to create tabix index: no index file produced")
            return False
    except Exception as e:
        logger.error(f"Error while creating tabix index: {str(e)}", exc_info=True)
        return False


def check_tabix_index(vcf_path: str):
    """Check whether the tabix index exists and is valid"""
    check_pysam_available("tabix index checking")

    if not vcf_path.endswith(".gz"):
        return False, "not a compressed VCF file"

    index_path = get_index_path(vcf_path)
    if not os.path.exists(index_path):
        return False, f"Index file does not exist: {index_path}"

    try:
        with pysam.TabixFile(vcf_path) as tbx:
            return True, "index is valid"
    except Exception as e:
        return False, f"index is invalid: {str(e)}"


def query_vcf_region(vcf_path: str, region: str) -> Optional[List[str]]:
    """Query a specific region of a VCF file using the tabix index.

    Records are materialized into a list before the ``TabixFile`` context
    closes, avoiding returning an iterator that is already invalid (once
    the pysam file handle is closed its iterator can no longer be used).
    """
    check_pysam_available("region query")

    if not vcf_path.endswith(".gz"):
        logger.error("Region query only supports compressed VCF files (.vcf.gz)")
        return None

    index_valid, msg = check_tabix_index(vcf_path)
    if not index_valid:
        logger.error(f"Index check failed: {msg}; trying to create the index automatically...")
        if not create_tabix_index(vcf_path, force=True):
            logger.error("Failed to create the index; cannot perform region query")
            return None

    try:
        with pysam.TabixFile(vcf_path) as tbx:
            return list(tbx.fetch(region=region))
    except ValueError as e:
        logger.error(f"Invalid region format {region}: {str(e)}")
        logger.info("Region format should be: chromosome:start-end (e.g., chr1:1000-5000)")
        return None
    except Exception as e:
        logger.error(f"Region query failed: {str(e)}", exc_info=True)
        return None


def compress_vcf_file(
    input_path: str, output_path: Optional[str] = None, force: bool = False
) -> bool:
    """Compress a VCF file to gzip format"""
    validate_file_exists(input_path, "VCF file")
    if output_path is None:
        output_path = input_path + ".gz"
    if os.path.exists(output_path) and not force:
        logger.error(f"Output file already exists: {output_path}; use --force to overwrite")
        return False

    try:
        logger.info(f"Compressing file: {input_path} -> {output_path}")
        if PYSAM_AVAILABLE:
            # pysam.tabix_compress produces BGZF format, which can be
            # indexed with tabix directly after compression
            pysam.tabix_compress(input_path, output_path)
        else:
            logger.warning(
                "pysam is not installed; falling back to plain gzip compression "
                "(the product cannot be indexed with tabix; "
                "recommended: pip install pysam and retry)"
            )
            with open(input_path, "rb") as f_in, gzip.open(
                output_path, "wb", compresslevel=6
            ) as f_out:
                shutil.copyfileobj(f_in, f_out, length=BLOCK_SIZE)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"File compressed successfully, size: {os.path.getsize(output_path):,} bytes")
            return True
        else:
            logger.error("File compression failed: no valid file produced")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    except Exception as e:
        logger.error(f"Error while compressing file: {str(e)}", exc_info=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def decompress_vcf_file(
    input_path: str, output_path: Optional[str] = None, force: bool = False
) -> bool:
    """Decompress a gzip-compressed VCF file"""
    if not input_path.endswith(".gz"):
        logger.error("Only .gz files can be decompressed")
        return False
    validate_file_exists(input_path, "compressed VCF file")
    if output_path is None:
        output_path = input_path[:-3]
    if os.path.exists(output_path) and not force:
        logger.error(f"Output file already exists: {output_path}; use --force to overwrite")
        return False

    try:
        logger.info(f"Decompressing file: {input_path} -> {output_path}")
        with gzip.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=BLOCK_SIZE)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"File decompressed successfully, size: {os.path.getsize(output_path):,} bytes")
            return True
        else:
            logger.error("File decompression failed: no valid file produced")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    except Exception as e:
        logger.error(f"Error while decompressing file: {str(e)}", exc_info=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def het_gt_to_missing(gt: str):
    """Convert heterozygous or half-missing genotypes to missing;
    returns (new GT, whether modified).

    Rules are consistent with het2missing.sh (bcftools +setGT):
    - heterozygous (0/1, 1/2, 0|1, etc., including multiallelic and
      phased heterozygotes) -> missing
    - half-missing (./1, .|0, etc.) -> normalized to fully missing
    - homozygous (0/0, 1/1, 2/2, etc.) and already missing (./.) are
      kept unchanged
    The separator (/ or |) is preserved.
    """
    if "|" in gt:
        sep = "|"
    elif "/" in gt:
        sep = "/"
    else:
        return gt, False  # no separator (e.g., haploid '.' or '0'); skip

    alleles = gt.split(sep)
    if any(a in (".", "") for a in alleles):
        new_gt = sep.join("." for _ in alleles)  # half-missing -> fully missing
    elif any(a != alleles[0] for a in alleles[1:]):
        new_gt = sep.join("." for _ in alleles)  # heterozygous
    else:
        return gt, False  # homozygous; keep

    return (new_gt, True) if new_gt != gt else (gt, False)


def is_bgzf_file(file_path: str) -> bool:
    """Determine whether a .gz file is in BGZF format (a prerequisite for
    creating a tabix index / region queries).

    BGZF carries a 'BC' subfield in the FEXTRA extra field of the gzip
    header, which is used here for identification. Plain gzip files (such
    as those produced by the Python gzip module) cannot be indexed with
    tabix.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(18)
    except OSError:
        return False
    if len(header) < 18 or header[:2] != b"\x1f\x8b":
        return False
    if not (header[3] & 0x04):  # FEXTRA flag bit
        return False
    return b"BC" in header[12:18]


class ProgressBar:
    """Lightweight terminal progress bar: percentage | rate | ETA.

    Rendered only when stderr is a terminal (not redirected/piped) and the
    total line count is known; stays silent otherwise. Does not introduce
    third-party dependencies such as tqdm.
    """

    def __init__(self, total: int, desc: str = "Processing", min_interval: float = 0.2):
        self.total = total
        self.desc = desc
        self.min_interval = min_interval
        self.n = 0
        self.start_time = time.time()
        self._last_render = 0.0
        # Prefer stderr; fall back to stdout if stderr is redirected
        # (render as long as either channel is a terminal)
        self._stream = sys.stderr if sys.stderr.isatty() else sys.stdout
        self.enabled = (
            total > 0
            and hasattr(self._stream, "isatty")
            and self._stream.isatty()
        )

    def update(self, n: int) -> None:
        self.n = n
        if not self.enabled:
            return
        now = time.time()
        if n < self.total and now - self._last_render < self.min_interval:
            return
        self._last_render = now
        self._render(now)

    def _render(self, now: float) -> None:
        elapsed = max(now - self.start_time, 1e-6)
        ratio = min(self.n / self.total, 1.0)
        bar_len = 30
        filled = int(bar_len * ratio)
        bar = "█" * filled + "░" * (bar_len - filled)
        rate = self.n / elapsed
        eta = (self.total - self.n) / rate if rate > 0 else 0
        self._stream.write(
            f"\r{self.desc} |{bar}| {ratio * 100:6.2f}% "
            f"({self.n:,}/{self.total:,}) {rate:,.0f} lines/s "
            f"ETA {timedelta(seconds=int(eta))}"
        )
        self._stream.flush()

    def finish(self) -> None:
        if not self.enabled:
            return
        self._render(time.time())
        self._stream.write("\n")
        self._stream.flush()


def run_preflight_checks(args) -> None:
    """Pre-flight checks: Python version, input file, dependencies,
    output directory and disk space.

    Any failed check logs an error and exits (exit code 1), avoiding
    failures halfway through processing.
    """
    # 1. Python version
    if sys.version_info < (3, 7):
        logger.error(f"Python >= 3.7 is required; current version is {platform.python_version()}")
        sys.exit(1)

    # 2. Input file exists, is non-empty and readable
    #    (validate_file_exists includes the permission check)
    validate_file_exists(args.vcf, "input VCF file")
    if os.path.getsize(args.vcf) == 0:
        logger.error(f"Input file is empty: {args.vcf}")
        sys.exit(1)
    if args.vcf.endswith(".gz"):
        with open(args.vcf, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                logger.error(f"File ends with .gz but is not a valid gzip file: {args.vcf}")
                sys.exit(1)

    # 3. Dependency and compression-format checks
    if (args.create_index or args.region) and not PYSAM_AVAILABLE:
        logger.error(f"This mode requires the pysam library; please install it first: pip install pysam")
        sys.exit(1)
    if args.vcf.endswith(".gz") and not is_bgzf_file(args.vcf):
        if args.create_index:
            logger.error(
                "Tabix indexing requires a BGZF-compressed (bgzip) .vcf.gz; "
                "this file is plain gzip."
            )
            logger.error("Please decompress it and recompress with bgzip first: bcftools view -Oz -o out.vcf.gz in.vcf")
            sys.exit(1)
        if args.region:
            logger.error(
                "Region query requires a BGZF-compressed (bgzip) .vcf.gz with its "
                "index; this file is plain gzip."
            )
            sys.exit(1)
        logger.warning(
            "Input is plain gzip rather than BGZF: streaming processing is "
            "unaffected, but no tabix index can be created and no region query "
            "can be performed on it"
        )

    # 4. Output directory and disk space
    out_dir = args.outdir or os.getcwd()
    validate_directory(out_dir)
    free_bytes = shutil.disk_usage(out_dir).free
    input_size = os.path.getsize(args.vcf)
    if free_bytes < input_size:
        logger.error(
            f"Insufficient disk space in the output directory: {free_bytes / 1e9:.1f} GB free "
            f"< input file size {input_size / 1e9:.1f} GB ({out_dir})"
        )
        sys.exit(1)
    if free_bytes < input_size * 2:
        logger.warning(
            f"Free disk space in the output directory is low ({free_bytes / 1e9:.1f} GB); "
            "it is recommended to reserve at least twice the input size"
        )

    # 5. Basic VCF header validation
    try:
        with open_file_smart(args.vcf, "r") as f:
            first_line = f.readline()
        if not first_line.startswith("##fileformat=VCF"):
            logger.error(
                f"File lacks the VCF header (##fileformat=VCF); it may not be a VCF file: {args.vcf}"
            )
            sys.exit(1)
    except (OSError, EOFError) as e:
        logger.error(f"Cannot read input file content: {e}")
        sys.exit(1)

    logger.info("Pre-flight checks passed ✓")


class VcfStatistics:
    """Collector class for VCF file statistics"""

    def __init__(self):
        self.total_variants = 0
        self.total_snps = 0
        self.total_indels = 0
        self.total_other_variants = 0
        self.total_monomorphic = 0
        self.total_polymorphic = 0
        self.chromosome_counts = defaultdict(int)
        self.variant_types = defaultdict(int)
        self.ref_allele_counts = Counter()
        self.alt_allele_counts = Counter()
        self.allele_count_distribution = defaultdict(int)
        self.quality_distribution = {
            "QUAL": defaultdict(int),
            "DP": defaultdict(int),
            "GQ": defaultdict(int),
        }
        self.quality_summary = {
            "QUAL": {"min": None, "max": None, "sum": 0, "count": 0, "mean": 0},
            "DP": {"min": None, "max": None, "sum": 0, "count": 0, "mean": 0},
            "GQ": {"min": None, "max": None, "sum": 0, "count": 0, "mean": 0},
        }
        self.genotype_counts = defaultdict(int)
        self.genotype_transitions = defaultdict(int)
        self.missing_genotypes = 0
        self.total_genotypes = 0
        self.filtered_variants = 0
        self.modified_genotypes = 0
        self.format_columns = []
        self.sample_names = []
        self.sample_count = 0
        self.info_fields = defaultdict(set)
        self.info_counts = defaultdict(int)
        self.position_ranges = defaultdict(lambda: {"min": None, "max": None})
        self.processing_info = {}

    def update_variant_type(self, ref: str, alt: str):
        self.total_variants += 1
        ref = ref.upper()
        alt_alleles = [a.upper() for a in alt.split(",") if a]

        # Monomorphic sites (ALT is "." or empty) are counted separately
        # and do not participate in the SNP/Indel classification
        if not alt_alleles or alt_alleles == ["."]:
            self.total_monomorphic += 1
            self.variant_types["Monomorphic"] += 1
            self.ref_allele_counts[ref] += 1
            self.allele_count_distribution[1] += 1
            return

        is_snp = all(
            len(a) == 1 and len(ref) == 1 and a in {"A", "T", "C", "G", "N"}
            for a in alt_alleles
        )
        is_indel = any(len(a) != len(ref) for a in alt_alleles)

        if is_snp and not is_indel:
            self.total_snps += 1
            self.variant_types["SNP"] += 1
            for a in alt_alleles:
                if {ref, a} in [{"A", "G"}, {"C", "T"}]:
                    self.genotype_transitions["transition"] += 1
                elif ref != a:
                    self.genotype_transitions["transversion"] += 1
        elif is_indel:
            self.total_indels += 1
            if len(ref) > max(len(a) for a in alt_alleles):
                self.variant_types["Deletion"] += 1
            elif len(ref) < min(len(a) for a in alt_alleles):
                self.variant_types["Insertion"] += 1
            else:
                self.variant_types["Indel"] += 1
        else:
            self.total_other_variants += 1
            self.variant_types["Other"] += 1

        self.ref_allele_counts[ref] += 1
        for a in alt_alleles:
            self.alt_allele_counts[a] += 1
        self.allele_count_distribution[1 + len(alt_alleles)] += 1
        self.total_polymorphic += 1

    def update_chromosome_stats(self, chrom: str, pos_str: str):
        self.chromosome_counts[chrom] += 1
        pos = int(pos_str)
        if (
            self.position_ranges[chrom]["min"] is None
            or pos < self.position_ranges[chrom]["min"]
        ):
            self.position_ranges[chrom]["min"] = pos
        if (
            self.position_ranges[chrom]["max"] is None
            or pos > self.position_ranges[chrom]["max"]
        ):
            self.position_ranges[chrom]["max"] = pos

    def update_quality_metrics(
        self, qual: str, info: str, format_data: list, sample_data: list
    ):
        try:
            qual_val = float(qual) if qual != "." else 0.0
            self._update_quality_summary("QUAL", qual_val)
        except ValueError:
            pass

        info_dict = {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in info.split(";")
            if "=" in item
        }
        if "DP" in info_dict:
            try:
                self._update_quality_summary("DP", int(info_dict["DP"]))
            except (ValueError, TypeError):
                pass

        if "GQ" in format_data:
            gq_index = format_data.index("GQ")
            for sample in sample_data:
                sample_fields = sample.split(":")
                if len(sample_fields) > gq_index and sample_fields[gq_index] != ".":
                    try:
                        self._update_quality_summary("GQ", int(sample_fields[gq_index]))
                    except ValueError:
                        pass

    def _update_quality_summary(self, metric: str, value: float):
        summary = self.quality_summary[metric]
        summary["sum"] += value
        summary["count"] += 1
        if summary["min"] is None or value < summary["min"]:
            summary["min"] = value
        if summary["max"] is None or value > summary["max"]:
            summary["max"] = value
        summary["mean"] = summary["sum"] / summary["count"]

        bin_size = 50 if metric == "QUAL" else 10
        value_bin = (int(value) // bin_size) * bin_size
        self.quality_distribution[metric][value_bin] += 1

    def update_genotype_stats(self, sample_data: list, format_data: list):
        if "GT" not in format_data:
            return
        gt_index = format_data.index("GT")
        for sample in sample_data:
            self.total_genotypes += 1
            gt = (
                sample.split(":")[gt_index]
                if len(sample.split(":")) > gt_index
                else "./."
            )
            if gt in ("./.", ".|."):
                self.genotype_counts["Missing"] += 1
                self.missing_genotypes += 1
            elif gt in ("0/0", "0|0"):
                self.genotype_counts["Homozygous Ref"] += 1
            elif re.match(r"^[1-9]/[1-9]$", gt) or re.match(r"^[1-9]\|[1-9]$", gt):
                self.genotype_counts["Homozygous Alt"] += 1
            elif re.match(r"^0[/|][1-9]$", gt) or re.match(r"^[1-9][/|]0$", gt):
                self.genotype_counts["Heterozygous"] += 1
            else:
                self.genotype_counts["Other"] += 1

    def update_info_fields(self, info: str):
        for field in info.split(";"):
            key = field.split("=", 1)[0]
            self.info_counts[key] += 1

    def record_format_and_samples(self, format_col: str, samples: list):
        self.format_columns = format_col.split(":")
        self.sample_names = samples
        self.sample_count = len(samples)

    def get_transition_transversion_ratio(self) -> float:
        t, tv = self.genotype_transitions.get(
            "transition", 0
        ), self.genotype_transitions.get("transversion", 0)
        return t / tv if tv > 0 else float("inf") if t > 0 else 0.0

    def get_missing_rate(self) -> float:
        return (
            self.missing_genotypes / self.total_genotypes
            if self.total_genotypes > 0
            else 0.0
        )

    def set_processing_info(self, **kwargs):
        self.processing_info.update(kwargs)

    def write_statistics_report(self, output_dir: str):
        validate_directory(output_dir)
        # Main text report
        report_path = os.path.join(output_dir, "vcf_statistics_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            self._write_main_report(f)

        # CSV data exports
        for metric, data in self.quality_distribution.items():
            csv_path = os.path.join(output_dir, f"{metric}_distribution.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["bin", "count"])
                writer.writerows(sorted(data.items()))

        # Generate the enhanced HTML summary report
        summary_path = self.generate_summary_report(output_dir)
        logger.info(f"Statistics report saved to: {report_path}")
        logger.info(f"Interactive HTML report saved to: {summary_path}")

    def generate_summary_report(self, output_dir: str) -> str:
        """Generates an enhanced, interactive HTML summary report with charts."""
        report_path = os.path.join(output_dir, "vcf_summary_report.html")

        # --- Prepare Data for Charts ---
        # 1. Variant Types Chart (Doughnut)
        variant_chart_labels = list(self.variant_types.keys())
        variant_chart_data = list(self.variant_types.values())

        # 2. Chromosome Distribution Chart (Bar) - Top 15
        sorted_chroms = sorted(
            self.chromosome_counts.items(), key=lambda item: item[1], reverse=True
        )
        chrom_chart_labels = [c[0] for c in sorted_chroms[:15]]
        chrom_chart_data = [c[1] for c in sorted_chroms[:15]]

        # --- HTML and CSS ---
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VCF Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f4f7f9; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; }}
        header {{ background-color: #fff; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        header h1 {{ margin: 0; color: #2c3e50; }}
        header p {{ margin: 5px 0 0; color: #7f8c8d; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
        .card {{ background-color: #fff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .card h2 {{ margin-top: 0; border-bottom: 2px solid #eef2f5; padding-bottom: 10px; color: #34495e; }}
        .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        .metric {{ text-align: center; background-color: #f8fafc; padding: 15px; border-radius: 6px; }}
        .metric .value {{ font-size: 2em; font-weight: 600; color: #2980b9; }}
        .metric .label {{ font-size: 0.9em; color: #95a5a6; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #eef2f5; }}
        th {{ color: #7f8c8d; font-weight: 500; }}
        td {{ font-weight: 500; }}
        .full-width {{ grid-column: 1 / -1; }}
        footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eef2f5; color: #95a5a6; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>VCF Analysis Report</h1>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="grid">
            <div class="card full-width">
                <h2>At a Glance</h2>
                <div class="summary-grid">
                    <div class="metric"><div class="value">{self.total_variants:,}</div><div class="label">Total Variants</div></div>
                    <div class="metric"><div class="value">{self.sample_count}</div><div class="label">Samples</div></div>
                    <div class="metric"><div class="value">{self.get_transition_transversion_ratio():.3f}</div><div class="label">Ti/Tv Ratio</div></div>
                    <div class="metric"><div class="value">{self.get_missing_rate():.2%}</div><div class="label">Missing Genotypes</div></div>
                </div>
            </div>

            <div class="card">
                <h2>Variant Types</h2>
                <canvas id="variantTypesChart"></canvas>
            </div>

            <div class="card">
                <h2>Genotype Distribution</h2>
                <table>
                    <thead><tr><th>Genotype</th><th>Count</th><th>Percentage</th></tr></thead>
                    <tbody>
                    {''.join(f"<tr><td>{k}</td><td>{v:,}</td><td>{v/self.total_genotypes*100:.2f}%</td></tr>" for k, v in self.genotype_counts.items() if self.total_genotypes > 0)}
                    </tbody>
                </table>
            </div>

            <div class="card full-width">
                <h2>Variant Distribution per Chromosome (Top 15)</h2>
                <canvas id="chromosomeChart"></canvas>
            </div>

            <div class="card full-width">
                <h2>Quality Summary</h2>
                <table>
                     <thead><tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th></tr></thead>
                     <tbody>
                     {''.join(f"<tr><td>{k}</td><td>{v['min']:.2f}</td><td>{v['max']:.2f}</td><td>{v['mean']:.2f}</td></tr>" for k, v in self.quality_summary.items() if v['count'] > 0)}
                     </tbody>
                </table>
            </div>
        </div>
        <footer>
            VCF Processor v{SCRIPT_VERSION} | A high-performance analysis tool
        </footer>
    </div>

<script>
document.addEventListener('DOMContentLoaded', function () {{
    // Chart 1: Variant Types (Doughnut)
    const variantCtx = document.getElementById('variantTypesChart').getContext('2d');
    new Chart(variantCtx, {{
        type: 'doughnut',
        data: {{
            labels: {variant_chart_labels},
            datasets: [{{
                label: 'Variant Count',
                data: {variant_chart_data},
                backgroundColor: ['#3498db', '#e74c3c', '#9b59b6', '#f1c40f', '#2ecc71'],
                hoverOffset: 4
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{ position: 'top' }},
                title: {{ display: false }}
            }}
        }}
    }});

    // Chart 2: Chromosome Distribution (Bar)
    const chromCtx = document.getElementById('chromosomeChart').getContext('2d');
    new Chart(chromCtx, {{
        type: 'bar',
        data: {{
            labels: {chrom_chart_labels},
            datasets: [{{
                label: 'Variant Count',
                data: {chrom_chart_data},
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }}]
        }},
        options: {{
            responsive: true,
            indexAxis: 'x',
            scales: {{ y: {{ beginAtZero: true }} }},
            plugins: {{
                legend: {{ display: false }},
                title: {{ display: false }}
            }}
        }}
    }});
}});
</script>
</body>
</html>
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        return report_path

    def _write_main_report(self, f):
        """Writes a more readable, formatted plain text report."""
        f.write("=" * 60 + "\n")
        f.write(" VCF STATISTICS REPORT\n")
        f.write("=" * 60 + "\n\n")

        info = self.processing_info
        f.write(f"{'Input File:':<25} {info.get('input_file', 'N/A')}\n")
        f.write(f"{'Report Generated:':<25} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 60 + "\n")

        f.write(f"{'SUMMARY':^60}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Total Variants:':<25} {self.total_variants:,}\n")
        f.write(f"{'Total Genotypes:':<25} {self.total_genotypes:,}\n")
        f.write(f"{'Samples:':<25} {self.sample_count:,}\n")
        f.write(f"{'Missing Genotype Rate:':<25} {self.get_missing_rate():.2%}\n")
        f.write(
            f"{'Ti/Tv Ratio:':<25} {self.get_transition_transversion_ratio():.3f}\n\n"
        )

        f.write("-" * 60 + "\n")
        f.write(f"{'VARIANT TYPES':^60}\n")
        f.write("-" * 60 + "\n")
        for v_type, count in self.variant_types.items():
            f.write(f"{v_type + ':':<25} {count:,} ({count/self.total_variants:.2%})\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write(f"{'QUALITY METRICS (MEAN)':^60}\n")
        f.write("-" * 60 + "\n")
        for metric, stats in self.quality_summary.items():
            if stats["count"] > 0:
                f.write(
                    f"{metric + ':':<25} {stats['mean']:.2f} (Min: {stats['min']:.2f}, Max: {stats['max']:.2f})\n"
                )
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write(" End of Report\n")
        f.write("=" * 60 + "\n")


class VcfFilter:
    """Filter rule class for VCF files"""

    def __init__(
        self, site_names: list, thresholds: list, min_quality: Optional[float] = None
    ):
        self.site_names = [name.strip().upper() for name in site_names]
        self.thresholds = [float(thresh) for thresh in thresholds]
        if len(self.site_names) != len(self.thresholds):
            logger.error(
                f"Filter names ({len(self.site_names)}) and thresholds ({len(self.thresholds)}) count mismatch."
            )
            sys.exit(1)
        self.min_quality = min_quality
        self.site_indices = {}

    def set_site_indices(self, format_columns: list):
        for name in self.site_names:
            if name not in format_columns:
                print (format_columns)
                logger.error(
                    f"Filter metric '{name}' not in VCF FORMAT column: {format_columns}"
                )
                sys.exit(1)
            self.site_indices[name] = format_columns.index(name)

    def filter_variant(self, qual: str) -> bool:
        if self.min_quality is not None:
            try:
                if float(qual) < self.min_quality:
                    return True
            except ValueError:
                return True
        return False

    def filter_genotype(self, sample_data: str) -> bool:
        sample_values = sample_data.split(":")
        try:
            for name, threshold in zip(self.site_names, self.thresholds):
                idx = self.site_indices[name]
                if (
                    idx >= len(sample_values)
                    or sample_values[idx] == "."
                    or float(sample_values[idx]) < threshold
                ):
                    return True
        except (IndexError, ValueError):
            return True
        return False


class VcfFileProcessor:
    """Main processor class for VCF files"""

    def __init__(
        self,
        vcf_path: str,
        out_dir: str,
        prefix: str,
        filters: Optional[VcfFilter],
        stats_only: bool,
        region: Optional[str],
        het2missing: bool = False,
        force: bool = False,
    ):
        validate_file_exists(vcf_path)
        self.input_vcf_path = os.path.abspath(vcf_path)
        self.is_compressed = self.input_vcf_path.endswith(".gz")
        self.region = region

        self.out_dir = out_dir if out_dir else os.getcwd()
        validate_directory(self.out_dir)

        base_prefix = (
            prefix
            if prefix
            else os.path.splitext(os.path.basename(self.input_vcf_path))[0].replace(
                ".vcf", ""
            )
        )
        if self.region:
            base_prefix += f".region_{re.sub(r'[:/-]', '_', self.region)}"

        self.stats = VcfStatistics()
        self.stats_only = stats_only
        self.filters = filters
        self.het2missing = het2missing
        self.filter_enabled = not stats_only and self.filters is not None

        self.stats_dir = os.path.join(self.out_dir, "statistics")
        if not stats_only:
            ext = ".vcf.gz" if self.is_compressed else ".vcf"
            if filters is not None:
                names = "_".join(filters.site_names)
                values = "_".join(str(t) for t in filters.thresholds)
                filter_suffix = f"_filtered.{names}.{values}"
            elif het2missing:
                filter_suffix = ".het2missing"
            else:
                filter_suffix = ""
            self.output_vcf_path = os.path.join(
                self.out_dir, f"{base_prefix}{filter_suffix}{ext}"
            )
            if os.path.exists(self.output_vcf_path) and not force:
                logger.error(
                    f"Output file already exists: {self.output_vcf_path}\n"
                    "To avoid accidental overwriting, please change the output "
                    "directory/prefix, or use --force to overwrite"
                )
                sys.exit(1)
            if os.path.exists(self.output_vcf_path):
                logger.warning(f"--force: will overwrite the existing output file {self.output_vcf_path}")
        else:
            self.output_vcf_path = None

        self.metadata = []
        self.header_processed = False
        self.total_lines = 0
        self.processed_lines = 0

    def _count_total_lines(self):
        if self.region:
            return
        logger.info("Counting total lines...")
        try:
            # Pure Python implementation: avoids relying on external commands
            # and eliminates shell injection risks.
            with open_file_smart(self.input_vcf_path, "r") as f:
                self.total_lines = sum(1 for _ in f)
            logger.info(f"File has {self.total_lines:,} lines.")
        except Exception as e:
            logger.warning(f"Could not count total lines: {e}")
            self.total_lines = 0

    def _process_header_line(self, line: str) -> bool:
        if line.startswith("##"):
            self.metadata.append(line)
            return True

        if line.startswith("#CHROM"):
            self.metadata.append(line)
            columns = line.strip().split()
            # The #CHROM header provides the sample names. The actual FORMAT keys
            # (like GT, DP, GQ) are defined in column 9 of each data line, not here.
            # We'll parse those from the first data line we encounter.
            sample_names = columns[9:]
            self.stats.sample_names = sample_names
            self.stats.sample_count = len(sample_names)
            self.header_processed = True
            return True
        return False

    def _process_data_line(self, line: str) -> Optional[str]:
        columns = line.strip().split("\t")
        # A VCF data line contains at least 8 fixed columns + the FORMAT column.
        if len(columns) < 9:
            logger.warning(
                f"Skipping malformed data line with {len(columns)} columns: "
                f"{line[:80]!r}"
            )
            return None

        # The first data line carries the real FORMAT keys (e.g., GT:AD:DP:GQ);
        # the filter is initialized based on them.
        if not self.stats.format_columns:
            format_keys = columns[8].split(':')
            self.stats.format_columns = format_keys
            if self.filter_enabled and self.filters:
                self.filters.set_site_indices(format_keys)

        chrom, pos, _, ref, alt, qual, _, info = columns[:8]
        fmt_col = columns[8]
        samples = columns[9:]

        self.stats.update_variant_type(ref, alt)
        self.stats.update_chromosome_stats(chrom, pos)
        self.stats.update_quality_metrics(qual, info, fmt_col.split(":"), samples)
        self.stats.update_genotype_stats(samples, fmt_col.split(":"))
        self.stats.update_info_fields(info)

        if self.stats_only:
            return None

        if self.filters and self.filters.filter_variant(qual):
            self.stats.filtered_variants += 1
            return None

        new_samples = []
        if self.filters or self.het2missing:
            fmt_keys = fmt_col.split(":")
            gt_index = (
                fmt_keys.index("GT")
                if self.het2missing and "GT" in fmt_keys
                else None
            )
            for sample in samples:
                sample_fields = sample.split(":")
                modified = False
                if self.filters and self.filters.filter_genotype(sample):
                    if sample_fields and sample_fields[0] not in (".", "./.", ".|"):
                        sample_fields[0] = "./."
                        modified = True
                if (
                    self.het2missing
                    and gt_index is not None
                    and len(sample_fields) > gt_index
                ):
                    new_gt, gt_changed = het_gt_to_missing(sample_fields[gt_index])
                    if gt_changed:
                        sample_fields[gt_index] = new_gt
                        modified = True
                if modified:
                    self.stats.modified_genotypes += 1
                    new_samples.append(":".join(sample_fields))
                else:
                    new_samples.append(sample)
            return "\t".join(columns[:9] + new_samples) + "\n"

        return line

    @timer_decorator
    def process(self) -> Dict[str, Any]:
        """Process the VCF file using a streaming approach."""
        self.start_time = time.time()
        self._count_total_lines()

        vcf_iterator = None
        if self.region:
            if not self.is_compressed or not PYSAM_AVAILABLE:
                logger.error(
                    "Region query requires a compressed (.vcf.gz) file and the 'pysam' library."
                )
                sys.exit(1)

            header_lines = []
            with open_file_smart(self.input_vcf_path, "r") as f_in:
                for line in f_in:
                    if line.startswith("#"):
                        header_lines.append(line)
                    else:
                        break

            records_iterator = query_vcf_region(self.input_vcf_path, self.region)
            if records_iterator is None:
                sys.exit(1)

            vcf_iterator = itertools.chain(header_lines, records_iterator)
        else:
            vcf_iterator = open_file_smart(self.input_vcf_path, "r")

        out_file = (
            open_file_smart(self.output_vcf_path, "w") if not self.stats_only else None
        )

        progress = ProgressBar(
            self.total_lines,
            desc="het2missing" if self.het2missing else "Processing",
        )

        try:
            start_time = time.time()
            last_report_time = start_time

            for line_num, line in enumerate(vcf_iterator, 1):
                self.processed_lines = line_num
                progress.update(line_num)

                if not self.header_processed:
                    is_header = self._process_header_line(line)
                    if out_file and is_header:
                        out_file.write(line)
                    if (
                        not is_header and line.strip()
                    ):  # Check for content to avoid errors on empty lines
                        logger.error("VCF header is missing or malformed.")
                        sys.exit(1)
                else:  # Already processed header, now process data lines
                    processed_line = self._process_data_line(line)
                    if out_file and processed_line:
                        out_file.write(processed_line)

                if time.time() - last_report_time >= 30:
                    self._report_progress(start_time)
                    last_report_time = time.time()

            self._report_progress(start_time, final=True)

        finally:
            progress.finish()
            if out_file:
                out_file.close()
                if self.output_vcf_path.endswith(".gz"):
                    if is_bgzf_file(self.output_vcf_path):
                        logger.info("Creating tabix index for output file...")
                        create_tabix_index(
                            self.output_vcf_path, force=True, fatal=False
                        )
                    else:
                        logger.warning(
                            "Output is plain gzip rather than BGZF; no tabix index "
                            "can be created. If an index is needed, recompress with "
                            "bgzip/bcftools and then run --create-index"
                        )
            if not self.region and hasattr(
                vcf_iterator, "close"
            ):  # Close file handle if it was opened
                vcf_iterator.close()

        filters_used_str = (
            ", ".join(
                [
                    f"{n}>{v}"
                    for n, v in zip(self.filters.site_names, self.filters.thresholds)
                ]
            )
            if self.filters
            else "None"
        )
        self.stats.set_processing_info(
            start_time=self.start_time,
            end_time=time.time(),
            input_file=self.input_vcf_path,
            output_file=self.output_vcf_path,
            filters_used=filters_used_str,
            region=self.region,
        )
        self.stats.write_statistics_report(self.stats_dir)

        logger.info("=" * 50 + "\nProcessing Complete!")
        return {"success": True}

    def _report_progress(self, start_time: float, final: bool = False):
        elapsed = time.time() - start_time
        progress_str = ""
        if self.total_lines > 0:
            progress = (self.processed_lines / self.total_lines) * 100
            progress_str = f"Progress: {progress:.1f}% ({self.processed_lines:,}/{self.total_lines:,}) | "

        rate = self.processed_lines / elapsed if elapsed > 0 else 0
        log_msg = (
            f"{'Final Stats' if final else 'Status'}: {progress_str}"
            f"Variants: {self.stats.total_variants:,} | "
            f"Lines/sec: {rate:,.0f} | Elapsed: {timedelta(seconds=int(elapsed))}"
        )
        logger.info(log_msg)


def main():
    """Main function to parse arguments and run the processor."""
    parser = argparse.ArgumentParser(
        description=f"Advanced VCF Processor v{SCRIPT_VERSION}. Filters, analyzes, and manages VCF files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--process", action="store_true", help="Process VCF file (default mode)"
    )
    mode_group.add_argument(
        "--compress", action="store_true", help="Compress VCF file to .vcf.gz"
    )
    mode_group.add_argument(
        "--decompress", action="store_true", help="Decompress .vcf.gz file"
    )
    mode_group.add_argument(
        "--create-index",
        action="store_true",
        help="Create tabix index for a .vcf.gz file",
    )
    mode_group.add_argument(
        "--het2missing",
        action="store_true",
        help="Convert heterozygous genotypes (0/1, 1/2, 0|1, ...) to missing; "
        "normalize half-missing (./1) to ./.; keep homozygous unchanged",
    )

    parser.add_argument(
        "-f", "--vcf", required=True, help="Input VCF file path (.vcf or .vcf.gz)"
    )
    parser.add_argument(
        "-o", "--output", help="Output file path (for compress/decompress modes)"
    )
    parser.add_argument(
        "-p", "--prefix", default="", help="Output file prefix (for process mode)"
    )
    parser.add_argument(
        "-d", "--outdir", default="", help="Output directory for results"
    )
    parser.add_argument(
        "-s",
        "--site_name",
        default="DP",
        help="Filter metric(s) from FORMAT column, comma-separated (e.g., DP,GQ)",
    )
    parser.add_argument(
        "-v",
        "--values",
        default="10",
        help="Threshold value(s) for filter metrics, comma-separated (e.g., 10,20)",
    )
    parser.add_argument(
        "-q",
        "--min_quality",
        type=float,
        help="Minimum variant quality (QUAL) to keep a site",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only generate statistics, do not filter or write new VCF",
    )
    parser.add_argument(
        "--region",
        help='Process a specific region, e.g., "chr1:1000-5000" (requires .vcf.gz + index)',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing output files or index",
    )

    args = parser.parse_args()

    run_start = time.time()
    out_dir = args.outdir or os.getcwd()
    log_path = setup_logging(out_dir)
    install_excepthook()
    log_run_banner(log_path, args)

    try:
        run_preflight_checks(args)

        if args.compress:
            compress_vcf_file(args.vcf, args.output, args.force)
        elif args.decompress:
            decompress_vcf_file(args.vcf, args.output, args.force)
        elif args.create_index:
            create_tabix_index(args.vcf, args.force)
        elif args.het2missing:
            processor = VcfFileProcessor(
                vcf_path=args.vcf,
                out_dir=args.outdir,
                prefix=args.prefix,
                filters=None,
                stats_only=False,
                region=args.region,
                het2missing=True,
                force=args.force,
            )
            processor.process()
            logger.info("Program finished successfully.")
        else:  # Default process mode
            filter_obj = None
            if not args.stats_only:
                filter_obj = VcfFilter(
                    site_names=args.site_name.split(","),
                    thresholds=args.values.split(","),
                    min_quality=args.min_quality,
                )

            processor = VcfFileProcessor(
                vcf_path=args.vcf,
                out_dir=args.outdir,
                prefix=args.prefix,
                filters=filter_obj,
                stats_only=args.stats_only,
                region=args.region,
                force=args.force,
            )
            processor.process()
            logger.info("Program finished successfully.")

        logger.info("=" * 70)
        logger.info(
            f"Run finished, total elapsed time: {timedelta(seconds=int(time.time() - run_start))}"
        )
        logger.info(f"Full run log: {log_path}")

    except SystemExit:
        # Exit from preflight / validation; leave a trace in the log
        logger.info(f"Program exited (see the error message above); log: {log_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        logger.error(f"Full run log: {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

