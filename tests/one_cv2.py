"""Fail unless each pip install report (from --dry-run --report) has exactly one distribution that provides cv2."""
import json
import sys

CV2_DISTRIBUTIONS = {
    "opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless",
}


def cv2_distributions(report: dict) -> list[str]:
    names = (item["metadata"]["name"].lower().replace("_", "-") for item in report.get("install", []))
    return sorted(name for name in names if name in CV2_DISTRIBUTIONS)


def main(paths: list[str]) -> int:
    failed = False
    for path in paths:
        with open(path, encoding="utf-8") as f:
            found = cv2_distributions(json.load(f))
        sys.stdout.write(f"{path}: {found}\n")
        failed |= len(found) != 1
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
