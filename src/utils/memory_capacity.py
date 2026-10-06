from pathlib import Path
import psutil


def available_memory():
    available = psutil.virtual_memory().available

    # Respect systemd/cgroup MemoryMax if one exists
    try:
        cgroup = next(
            line.split("::", 1)[1].strip()
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if "::" in line
        )

        base = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        maximum = (base / "memory.max").read_text().strip()

        if maximum != "max":
            current = int((base / "memory.current").read_text())
            available = min(available, int(maximum) - current)
    except (OSError, StopIteration, ValueError):
        pass

    return max(0, available)


def memory_capacity():
    capacity = psutil.virtual_memory().total

    # Respect systemd/cgroup MemoryMax if one exists
    try:
        cgroup = next(
            line.split("::", 1)[1].strip()
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if "::" in line
        )

        base = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        maximum = (base / "memory.max").read_text().strip()

        if maximum != "max":
            capacity = min(capacity, int(maximum))

    except (OSError, StopIteration, ValueError):
        pass

    return capacity



def guard_memory(required, operation, safety=0.8):
    available = available_memory()
    safe_available = safety * available

    if required > safe_available:
        raise MemoryError(
            f"{operation} would require approximately "
            f"{required / 2**30:.2f} GiB, but only "
            f"{available / 2**30:.2f} GiB remain."
        )


def guard_remaining_memory(operation, reserve_percent=15):
    available = available_memory()
    capacity = memory_capacity()

    remaining_percent = 100 * available / capacity

    if remaining_percent < reserve_percent:
        raise MemoryError(
            f"Stopping {operation}: only "
            f"{available / 2**30:.2f} GiB remain "
            f"({remaining_percent:.1f}% of available capacity). "
            f"At least {reserve_percent}% must remain."
        )