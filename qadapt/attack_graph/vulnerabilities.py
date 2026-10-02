"""Vulnerability catalog and NVD (CVE API 2.0) record parsing.

The built-in catalog lists widely documented CVEs with their NVD CVSS v3 base
scores. It is meant to seed *synthetic* enterprise scenarios; refresh the
records from the NVD (``parse_nvd_record`` / ``fetch_nvd_cve``) before using
scores in published results.
"""

from __future__ import annotations

import json
import urllib.request

from qadapt.core.models import AssetType, Vulnerability

# cve_id, cvss_base, exploitability (0..1), requires_privileges, description
_CATALOG: list[tuple[str, float, float, bool, str]] = [
    ("CVE-2021-44228", 10.0, 0.97, False, "Apache Log4j2 JNDI RCE (Log4Shell)"),
    ("CVE-2022-22965", 9.8, 0.85, False, "Spring Framework RCE (Spring4Shell)"),
    ("CVE-2021-26855", 9.8, 0.90, False, "Exchange Server SSRF (ProxyLogon)"),
    ("CVE-2019-19781", 9.8, 0.90, False, "Citrix ADC path traversal RCE"),
    ("CVE-2018-13379", 9.8, 0.88, False, "FortiOS SSL-VPN path traversal"),
    ("CVE-2023-34362", 9.8, 0.85, False, "MOVEit Transfer SQL injection"),
    ("CVE-2014-0160", 7.5, 0.80, False, "OpenSSL Heartbleed information disclosure"),
    ("CVE-2017-0144", 8.1, 0.75, False, "SMBv1 RCE (EternalBlue)"),
    ("CVE-2019-0708", 9.8, 0.70, False, "RDP RCE (BlueKeep)"),
    ("CVE-2021-34527", 8.8, 0.65, True, "Windows Print Spooler RCE (PrintNightmare)"),
    ("CVE-2020-1472", 10.0, 0.80, False, "Netlogon privilege escalation (Zerologon)"),
    ("CVE-2021-3156", 7.8, 0.70, True, "sudo heap overflow privilege escalation (Baron Samedit)"),
    ("CVE-2012-2122", 5.1, 0.55, False, "MySQL authentication bypass"),
    ("CVE-2016-6662", 9.8, 0.50, True, "MySQL config-file RCE"),
]

CATALOG: dict[str, Vulnerability] = {
    c: Vulnerability(c, s, e, p, d) for c, s, e, p, d in _CATALOG
}

# Which catalog entries are plausible on which asset class.
APPLICABLE: dict[AssetType, tuple[str, ...]] = {
    AssetType.WEB_SERVER: ("CVE-2021-44228", "CVE-2022-22965", "CVE-2014-0160", "CVE-2023-34362"),
    AssetType.APP_SERVER: ("CVE-2021-44228", "CVE-2022-22965", "CVE-2021-3156"),
    AssetType.MAIL_SERVER: ("CVE-2021-26855",),
    AssetType.VPN_GATEWAY: ("CVE-2018-13379", "CVE-2019-19781"),
    AssetType.FIREWALL: ("CVE-2018-13379",),
    AssetType.DATABASE: ("CVE-2012-2122", "CVE-2016-6662"),
    AssetType.FILE_SERVER: ("CVE-2017-0144", "CVE-2021-34527"),
    AssetType.WORKSTATION: ("CVE-2017-0144", "CVE-2019-0708", "CVE-2021-34527"),
    AssetType.DOMAIN_CONTROLLER: ("CVE-2020-1472", "CVE-2021-34527"),
    AssetType.ADMIN: ("CVE-2021-3156",),
}


def parse_nvd_record(item: dict) -> Vulnerability:
    """Parse one ``vulnerabilities[i]`` entry of the NVD CVE API 2.0 response."""
    cve = item.get("cve", item)
    metrics = cve.get("metrics", {})
    base, expl, priv = 5.0, 0.5, False
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            m = metrics[key][0]
            data = m.get("cvssData", {})
            base = float(data.get("baseScore", base))
            # exploitability sub-score is 0..3.9 (v3) or 0..10 (v2)
            sub = float(m.get("exploitabilityScore", 0.0))
            expl = min(1.0, sub / (3.9 if key != "cvssMetricV2" else 10.0)) if sub else base / 10
            priv = data.get("privilegesRequired", "NONE") not in ("NONE", None)
            break
    desc = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")
    return Vulnerability(cve.get("id", "UNKNOWN"), base, expl, priv, desc[:200])


def fetch_nvd_cve(cve_id: str, timeout: float = 20.0) -> Vulnerability:  # pragma: no cover
    """Fetch a single CVE from the NVD API (network access required)."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.load(resp)
    return parse_nvd_record(payload["vulnerabilities"][0])
