def make_session_id(
    device_id: str,
    boot_id: int,
) -> str:
    safe_device = (
        device_id
        .replace(" ", "-")
        .replace("/", "-")
    )

    return f"{safe_device}-{boot_id:016x}"
