"""Translate domain definitions without leaking engine handles into the SDK."""

from xml.etree import ElementTree as ET

from edge_sim_models import ScenarioSpec


def platform_xml(scenario: ScenarioSpec) -> str:
    root = ET.Element("platform", version="4.1")
    zone = ET.SubElement(root, "zone", id="platform", routing="Full")
    for node in scenario.nodes:
        ET.SubElement(zone, "host", id=node.id, speed=str(node.speed_flops), core=str(node.cores))
    for link in scenario.links:
        ET.SubElement(
            zone,
            "link",
            id=link.id,
            bandwidth=str(link.bandwidth_bytes_s),
            latency=str(link.latency_s),
            sharing_policy="SHARED",
        )
    for route in scenario.routes:
        element = ET.SubElement(zone, "route", src=route.src, dst=route.dst, symmetrical="NO")
        for link_id in route.links:
            ET.SubElement(element, "link_ctn", id=link_id)
    return (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">\n'
        + ET.tostring(root, encoding="unicode")
    )
