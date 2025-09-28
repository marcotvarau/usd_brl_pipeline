"""
Notification Manager
Handles email and other notification services
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
import os


class NotificationManager:
    """
    Manager for sending notifications via email and other channels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize notification manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get notification configuration
        notifications_config = config.get('monitoring', {}).get('notifications', {})
        
        # Email configuration
        self.email_config = notifications_config.get('email', {})
        self.email_enabled = self.email_config.get('enabled', False)
        
        # Slack configuration (placeholder)
        self.slack_config = notifications_config.get('slack', {})
        self.slack_enabled = self.slack_config.get('enabled', False)
    
    def send_email(self, recipient: str, subject: str, message: str) -> bool:
        """
        Send email notification.
        
        Args:
            recipient: Email recipient
            subject: Email subject
            message: Email message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.email_enabled:
            self.logger.info("Email notifications disabled")
            return False
        
        try:
            # Get email configuration from environment or config
            smtp_server = os.getenv('SMTP_SERVER', self.email_config.get('smtp_server'))
            smtp_port = int(os.getenv('SMTP_PORT', self.email_config.get('smtp_port', 587)))
            sender = os.getenv('EMAIL_SENDER', self.email_config.get('sender'))
            password = os.getenv('EMAIL_PASSWORD', self.email_config.get('password'))
            
            if not all([smtp_server, sender, password]):
                self.logger.error("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def send_default(self, subject: str, message: str) -> bool:
        """
        Send notification to default recipients.
        
        Args:
            subject: Notification subject
            message: Notification message
            
        Returns:
            True if successful, False otherwise
        """
        recipients = self.email_config.get('recipients', [])
        
        if not recipients:
            self.logger.warning("No default recipients configured")
            return False
        
        success_count = 0
        for recipient in recipients:
            if self.send_email(recipient, subject, message):
                success_count += 1
        
        return success_count > 0
    
    def send_slack(self, message: str) -> bool:
        """
        Send Slack notification (placeholder implementation).
        
        Args:
            message: Slack message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.slack_enabled:
            self.logger.info("Slack notifications disabled")
            return False
        
        self.logger.warning("Slack notifications not implemented yet")
        return False
