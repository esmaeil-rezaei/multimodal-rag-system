# HIPAA Security Implementation Best Practices

## Framework Alignment
NIST Cybersecurity Framework maps well to HIPAA Security Rule requirements:

| NIST Function | HIPAA Mapping | Key Controls |
|--------------|--------------|-------------|
| Identify | Risk Analysis, Asset Management | Inventory, risk register |
| Protect | Administrative/Technical Safeguards | Access controls, encryption |
| Detect | Audit Controls, SIEM | Log monitoring, alerting |
| Respond | Incident Response | Playbooks, containment |
| Recover | Contingency Planning | BCP, DRP, data backups |

## Encryption Standards
| Data State | Minimum Standard | Recommended |
|-----------|-----------------|------------|
| Data at Rest | AES-128 | AES-256 |
| Data in Transit | TLS 1.2 | TLS 1.3 |
| Portable Media | AES-128 | AES-256 |
| Email with PHI | TLS 1.2 | S/MIME or PGP |
| Backups | AES-128 | AES-256 with offsite keys |

## Access Control Best Practices
1. **Role-Based Access Control (RBAC)** — Assign permissions based on job function
2. **Principle of Least Privilege** — Grant minimum access required
3. **Multi-Factor Authentication (MFA)** — Require for all remote access
4. **Privileged Access Management (PAM)** — Manage and monitor admin accounts
5. **Automated Provisioning/Deprovisioning** — Remove access within 24h of termination

## Audit Logging Requirements
Logs must capture for all systems containing ePHI:
- User authentication events (success and failure)
- Access to PHI records
- Modifications to PHI
- System events affecting PHI availability
- Administrative actions

Retention: Minimum 6 years (aligns with HIPAA document retention)

## Vendor/Cloud Considerations
| Control Area | On-Premise | Cloud (IaaS) | Cloud (SaaS) |
|-------------|-----------|-------------|-------------|
| Physical security | CE responsibility | Vendor | Vendor |
| Infrastructure security | CE | Shared | Vendor |
| Application security | CE | CE | Shared |
| Data security | CE | CE | Shared |
| BAA required | N/A | Yes | Yes |

## Annual Security Program Activities
| Activity | Frequency | Owner |
|----------|-----------|-------|
| Risk Analysis | Annual minimum | CISO/Privacy Officer |
| Penetration Testing | Annual | Security Team/Vendor |
| Security Awareness Training | Annual (new hires at onboarding) | HR/Security |
| Business Continuity Testing | Annual | IT/Operations |
| Access Review | Semi-annual | Department Managers |
| BA Agreement Review | Annual | Legal/Compliance |
| Patch Management Review | Monthly | IT |
| Log Review | Continuous/Weekly | Security Operations |

## Security Maturity Model
| Level | Description | Key Indicators |
|-------|-------------|---------------|
| 1 - Initial | Ad hoc, reactive | No formal program |
| 2 - Developing | Basic controls | Risk analysis done, some policies |
| 3 - Defined | Documented processes | Full policy suite, regular training |
| 4 - Managed | Measured and controlled | Metrics, audit results tracked |
| 5 - Optimizing | Continuous improvement | Threat intelligence, automation |
