output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.workstation.id
}

output "public_ip" {
  description = "Public IP address (connect via https://<ip>:8443 for DCV)"
  value       = aws_instance.workstation.public_ip
}

output "dcv_url" {
  description = "DCV remote desktop URL"
  value       = "https://${aws_instance.workstation.public_ip}:8443"
}

output "ssh_command" {
  description = "SSH command"
  value       = var.key_name != "" ? "ssh -i ${var.key_name}.pem ubuntu@${aws_instance.workstation.public_ip}" : "Use SSM Session Manager"
}
