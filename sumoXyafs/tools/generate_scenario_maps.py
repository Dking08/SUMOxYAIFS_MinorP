"""Generate publication-grade line/mesh road network maps for all three evaluated scenarios:
1. Delhi Arterial Corridor (India)
2. KIIT University Smart Campus (Bhubaneswar, India)
3. Luxembourg LuST Benchmark (Luxembourg)
Overlaid with RSU edge server deployment anchors, coverage zones, and lateral backhaul links.
"""
import json
import math
import os
import gzip
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
import sumolib

def load_network_shapes(net_path):
    """Extract lane shapes and types from a SUMO .net.xml or .net.xml.gz."""
    print(f"Loading network: {net_path}")
    net = sumolib.net.readNet(net_path)
    
    major_lines = []
    minor_lines = []
    
    for edge in net.getEdges():
        speed = edge.getSpeed()
        lanes = edge.getLanes()
        for lane in lanes:
            shape = lane.getShape()
            if len(shape) >= 2:
                # Classify into major and minor roads based on speed / lane count
                if speed >= 13.8 or len(lanes) >= 2:  # >= 50 km/h or multi-lane
                    major_lines.append(shape)
                else:
                    minor_lines.append(shape)
                    
    bbox = net.getBBoxXY()
    print(f"  Loaded {len(major_lines)} major lane segments, {len(minor_lines)} minor lane segments. BBox: {bbox}")
    return major_lines, minor_lines, bbox

def plot_single_scenario(ax, net_path, rsu_path, title, label_prefix, scale_km=True):
    major_lines, minor_lines, bbox = load_network_shapes(net_path)
    
    # Plot road network lines (clean line/mesh style)
    # Minor roads
    lc_minor = LineCollection(minor_lines, colors='#64748B', linewidths=0.6, alpha=0.7, zorder=2)
    ax.add_collection(lc_minor)
    
    # Major roads
    lc_major = LineCollection(major_lines, colors='#0F172A', linewidths=1.2, alpha=0.9, zorder=3)
    ax.add_collection(lc_major)
    
    # Load RSUs
    with open(rsu_path, 'r', encoding='utf-8') as f:
        rsu_cfg = json.load(f)
    rsus = rsu_cfg.get('rsus', [])
    
    # Draw lateral backhaul links between neighbor RSUs
    rsu_dict = {r['id']: (float(r['x']), float(r['y'])) for r in rsus}
    drawn_links = set()
    for r in rsus:
        r_id = r['id']
        x1, y1 = rsu_dict[r_id]
        for neighbor_id in r.get('neighbors', []):
            if neighbor_id in rsu_dict:
                x2, y2 = rsu_dict[neighbor_id]
                link_key = tuple(sorted([r_id, neighbor_id]))
                if link_key not in drawn_links:
                    drawn_links.add(link_key)
                    ax.plot([x1, x2], [y1, y2], color='#0284C7', linestyle='--', linewidth=1.4, alpha=0.85, zorder=4)
    
    # Draw RSU coverage circles and anchor dots
    cov_radius = 500.0  # meters
    for i, r in enumerate(rsus):
        x = float(r['x'])
        y = float(r['y'])
        
        # Coverage circle
        circ = Circle((x, y), cov_radius, facecolor='#E0F2FE', edgecolor='#0284C7', linestyle=':', linewidth=1.0, alpha=0.35, zorder=4)
        ax.add_patch(circ)
        
        # RSU Center Dot
        ax.scatter(x, y, color='#DC2626', s=45, edgecolor='#FFFFFF', linewidth=1.2, zorder=6)
        
        # RSU label tag
        short_id = f"R{i+1}"
        ax.text(x, y + 120, short_id, fontsize=7.5, fontweight='bold', ha='center', va='bottom',
                color='#991B1B', bbox=dict(boxstyle='round,pad=0.18', facecolor='#FEF2F2', edgecolor='#DC2626', linewidth=0.6), zorder=7)
        
    ax.set_xlim(bbox[0][0] - 200, bbox[1][0] + 200)
    ax.set_ylim(bbox[0][1] - 200, bbox[1][1] + 200)
    ax.set_aspect('equal', adjustable='box')
    
    # Formatting
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8, color='#0F172A')
    
    # Convert axes ticks to kilometers for clean presentation
    if scale_km:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, tick_number: f"{val/1000:.1f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, tick_number: f"{val/1000:.1f}"))
        ax.set_xlabel("East-West Distance (km)", fontsize=8, color='#334155')
        ax.set_ylabel("North-South Distance (km)", fontsize=8, color='#334155')
    else:
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        
    ax.tick_params(axis='both', which='major', labelsize=7.5)
    ax.grid(True, linestyle=':', alpha=0.4, color='#CBD5E1')
    ax.set_facecolor('#FAFAFA')

def generate_all_maps():
    out_dir = os.path.abspath("conference_paper/figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Individual high-res maps
    scenarios = [
        ("delhi", "sumoXyafs/sumo/delhi/osm.net.xml.gz", "sumoXyafs/rsu_config.json", 
         "Delhi Arterial Highway Corridor (9 RSUs)", "delhi"),
        ("kiit", "sumoXyafs/sumo/kiit_campus/kiit_campus.net.xml", "sumoXyafs/sumo/kiit_campus/rsu_config_kiit.json", 
         "KIIT Smart Campus Micro-Mobility Grid (8 RSUs)", "kiit"),
        ("lust", "sumoXyafs/sumo/lust/scenario/lust.net.xml", "sumoXyafs/sumo/lust/rsu_config_lust.json", 
         "Luxembourg LuST Metropolitan Network (12 RSUs)", "lust")
    ]
    
    for tag, net_p, rsu_p, title, label in scenarios:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        plot_single_scenario(ax, net_p, rsu_p, title, label)
        fig.tight_layout()
        out_file = os.path.join(out_dir, f"map_{tag}.png")
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] Saved {out_file}")
        
    # 2. Consolidated 3-Panel Figure for IEEE paper (fits 2-column or 1-column layout)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), dpi=300, gridspec_kw={'width_ratios': [0.65, 1.05, 1.15]})
    
    plot_single_scenario(axes[0], scenarios[0][1], scenarios[0][2], "(a) Delhi Arterial Corridor (India)\n$M=9$ RSUs, 4.2 km Linear Highway", "delhi")
    plot_single_scenario(axes[1], scenarios[1][1], scenarios[1][2], "(b) KIIT Smart Campus (Bhubaneswar, India)\n$M=8$ RSUs, 2.4 $\\text{km}^2$ Campus Area", "kiit")
    plot_single_scenario(axes[2], scenarios[2][1], scenarios[2][2], "(c) Luxembourg LuST Benchmark\n$M=12$ RSUs, 155 $\\text{km}^2$ Metropolitan Grid", "lust")
    
    # Common Legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#0F172A', lw=2.0, label='Major Road / Arterial'),
        Line2D([0], [0], color='#64748B', lw=1.0, label='Secondary Street / Lane'),
        Line2D([0], [0], color='#0284C7', lw=1.5, linestyle='--', label='Lateral RSU Backhaul Link'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC2626', markersize=8, label='RSU Edge Server Anchor'),
        Patch(facecolor='#E0F2FE', edgecolor='#0284C7', linestyle=':', label='V2I Coverage Cell ($R_{\\text{cov}}=500\\,$m)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8.5, frameon=True, 
               facecolor='#FFFFFF', edgecolor='#CBD5E1', bbox_to_anchor=(0.5, -0.05))
    
    fig.tight_layout()
    consolidated_file = os.path.join(out_dir, "00_scenario_topologies.png")
    fig.savefig(consolidated_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Successfully saved consolidated figure: {consolidated_file}")

if __name__ == '__main__':
    generate_all_maps()
