import json


def get_current_detection(camera_id):
    """Get the current detection in a given location"""
    detection_info = {
        "location": "(0, 0.4m)",
        "count": "5",
        "category": "coffee beans",
        "area": "0.2m^2",
    }
    return json.dumps(detection_info)


def get_current_task(category):
    """Get the current task in a given location"""
    task_info = {
        "type": "gathering",
        "target_radius": "0.1m",
        "target_center": "(0, 0)",
    }
    return json.dumps(task_info)
