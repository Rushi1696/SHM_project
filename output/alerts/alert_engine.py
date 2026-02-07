"""Alert engine module."""


def generate_alert(event):
    """Generate alert from event"""
    if event is None:
        return None
    return {
        'alert_id': event.get('id'),
        'severity': 'info',
        'message': f"Event: {event.get('type', 'unknown')}"
    }


def send_notification(alert):
    """Send notification for alert"""
    if alert:
        print(f"Alert: {alert.get('message')}")
        return True
    return False
