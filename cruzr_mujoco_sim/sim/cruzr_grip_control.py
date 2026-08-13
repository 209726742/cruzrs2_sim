"""Host-testable grip command helpers for CRUZR PGC teleop."""


def grip_open_fraction(raw_value, open_value, close_value):
    """Normalize a PGC raw finger command/position to 1=open, 0=closed."""
    span = float(open_value) - float(close_value)
    if abs(span) < 1e-9:
        return 0.0
    frac = (float(raw_value) - float(close_value)) / span
    return float(max(0.0, min(1.0, frac)))


def grip_command_after_goal_change(
    current_command,
    new_goal,
    open_value,
    close_value,
    soft_enabled,
):
    """Return the immediate command after a user changes the grip goal.

    With soft grip enabled, closing must not jump the command to full close in
    the same frame.  The runtime soft-grip limiter advances it using measured
    finger position, which caps penetration preload.  Opening can release
    immediately.
    """
    if not soft_enabled:
        return float(new_goal)
    if close_value <= open_value:
        return float(new_goal)
    if float(new_goal) <= float(current_command):
        return float(new_goal)
    return float(current_command)
