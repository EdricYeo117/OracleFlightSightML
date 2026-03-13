def vector_to_dir(dx, dy, thresh=0.10):
    h = "center"
    v = "center"

    if dx <= -thresh:
        h = "left"
    elif dx >= thresh:
        h = "right"

    if dy <= -thresh:
        v = "up"
    elif dy >= thresh:
        v = "down"

    return f"{v}-{h}"


def fmt_vec3(name, vec):
    if vec is None:
        return f"{name}=None"
    return f"{name}=({vec[0]:+.3f},{vec[1]:+.3f},{vec[2]:+.3f})"


def fmt_vec2(name, vec):
    if vec is None:
        return f"{name}=None"
    return f"{name}=({vec[0]:+.3f},{vec[1]:+.3f})"


def fmt_head(head):
    if head is None:
        return "head=None"
    return f"head=(yaw={head['yaw']:+.2f},pitch={head['pitch']:+.2f},roll={head['roll']:+.2f})"


def log_snapshot(frame_idx, left_gaze_vec, right_gaze_vec, tracker_snapshot, every_n=5):
    if frame_idx % every_n != 0:
        return

    final_vec = tracker_snapshot["final_vec"]
    final_dir = "unknown"
    if final_vec is not None:
        final_dir = vector_to_dir(final_vec[0], final_vec[1])

    print(
        f"[{frame_idx:05d}] "
        f"{fmt_vec3('left3d', left_gaze_vec)} "
        f"{fmt_vec3('right3d', right_gaze_vec)} "
        f"{fmt_vec2('iris', tracker_snapshot['iris_vec'])} "
        f"{fmt_vec2('l2cs', tracker_snapshot['l2cs_vec'])} "
        f"{fmt_vec2('final', tracker_snapshot['final_vec'])} "
        f"{fmt_head(tracker_snapshot['head_pose'])} "
        f"blink={tracker_snapshot['blink_like']} "
        f"dir={final_dir}"
    )