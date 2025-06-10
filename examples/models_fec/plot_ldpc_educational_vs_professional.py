"""
====================================================================
Educational vs Professional LDPC Codes: Understanding the Differences
====================================================================

This example demonstrates the fundamental differences between educational
hand-crafted LDPC codes :cite:`gallager1962low` and professional RPTU database codes, explaining
why direct performance comparison is inappropriate and misleading.

We focus on:
- Educational value of small hand-crafted codes
- Real-world applications of RPTU database codes
- Why these serve different purposes
- Proper evaluation methodologies for each type

**Key Message**: These code types serve different purposes and should not
be directly compared for "performance" - it's like comparing a bicycle
to an airplane for transportation efficiency.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from kaira.channels.analog import AWGNChannel
from kaira.metrics.signal import BitErrorRate
from kaira.models.fec.decoders import BeliefPropagationDecoder
from kaira.models.fec.encoders import LDPCCodeEncoder

# %%
# Configuration and Setup
# --------------------------------------
torch.manual_seed(42)
np.random.seed(42)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

print("Educational vs Professional LDPC Codes Analysis")
print("=" * 60)

# %%
# Educational Hand-crafted LDPC Codes
# --------------------------------------
# Small codes designed for learning and understanding


def create_educational_codes():
    """Create small LDPC codes perfect for education and visualization."""

    educational_codes = {}

    print("\n📚 EDUCATIONAL HAND-CRAFTED CODES")
    print("-" * 40)

    # Simple (6,3) regular LDPC code
    H1 = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

    educational_codes["Educational (6,3)"] = {
        "parity_check_matrix": H1,
        "name": "Educational LDPC (6,3)",
        "description": "Simple regular code for learning LDPC fundamentals",
        "n": 6,
        "k": 3,
        "rate": 0.5,
        "color": "#1f77b4",
        "purpose": "Understanding parity check matrices and BP decoding",
        "advantages": ["Easy to visualize", "Simple to analyze", "Fast simulation"],
        "use_cases": ["Education", "Algorithm development", "Proof of concept"],
    }

    # Slightly larger (8,3) code for more complex examples
    H2 = torch.tensor([[1, 1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.float32)

    educational_codes["Educational (8,3)"] = {
        "parity_check_matrix": H2,
        "name": "Educational LDPC (8,3)",
        "description": "Lower rate code demonstrating protection vs efficiency",
        "n": 8,
        "k": 3,
        "rate": 3 / 8,
        "color": "#ff7f0e",
        "purpose": "Demonstrating rate vs protection trade-offs",
        "advantages": ["More parity protection", "Complex enough for analysis"],
        "use_cases": ["Teaching error correction", "Rate comparison studies"],
    }

    return educational_codes


def create_professional_codes():
    """Load professional LDPC codes from RPTU database."""

    professional_codes = {}

    print("\n🏭 PROFESSIONAL RPTU DATABASE CODES")
    print("-" * 40)

    # WiMAX standard codes
    try:
        rptu_encoder_1 = LDPCCodeEncoder(rptu_database=True, code_length=576, code_dimension=288, rptu_standart="wimax")
        professional_codes["WiMAX (576,288)"] = {
            "encoder": rptu_encoder_1,
            "name": "WiMAX LDPC (576,288)",
            "description": "IEEE 802.16 WiMAX standard LDPC code",
            "n": 576,
            "k": 288,
            "rate": 0.5,
            "color": "#2ca02c",
            "standard": "IEEE 802.16 WiMAX",
            "purpose": "Wireless broadband communication",
            "advantages": ["Standards compliant", "Optimized for real channels", "Field proven"],
            "use_cases": ["WiMAX systems", "Production deployment", "Research baseline"],
        }
        print("✓ Loaded WiMAX (576,288) professional code")
    except Exception as e:
        print(f"⚠ Could not load WiMAX code: {e}")

    try:
        rptu_encoder_2 = LDPCCodeEncoder(rptu_database=True, code_length=672, code_dimension=336, rptu_standart="wigig")
        professional_codes["WiGig (672,336)"] = {
            "encoder": rptu_encoder_2,
            "name": "WiGig LDPC (672,336)",
            "description": "IEEE 802.11ad WiGig standard LDPC code",
            "n": 672,
            "k": 336,
            "rate": 0.5,
            "color": "#9467bd",
            "standard": "IEEE 802.11ad WiGig",
            "purpose": "Ultra-high-speed wireless communication",
            "advantages": ["Gigabit wireless", "Low latency", "Robust to interference"],
            "use_cases": ["60 GHz wireless", "High-throughput applications", "Standards testing"],
        }
        print("✓ Loaded WiGig (672,336) professional code")
    except Exception as e:
        print(f"⚠ Could not load WiGig code: {e}")

    return professional_codes


# Create the codes
educational_codes = create_educational_codes()
professional_codes = create_professional_codes()

# %%
# Visualization: Understanding Code Structure Differences
# --------------------------------------------------------

fig = plt.figure(figsize=(16, 10))

# Plot 1: Educational code matrix visualization
plt.subplot(2, 3, 1)
H_edu = educational_codes["Educational (6,3)"]["parity_check_matrix"]
plt.imshow(H_edu, cmap="RdYlBu_r", interpolation="nearest", aspect="auto")
for i in range(H_edu.shape[0]):
    for j in range(H_edu.shape[1]):
        plt.text(j, i, int(H_edu[i, j]), ha="center", va="center", color="white" if H_edu[i, j] == 1 else "black", fontweight="bold")
plt.title("Educational (6,3)\nParity Check Matrix", fontsize=11)
plt.xlabel("Variable Nodes")
plt.ylabel("Check Nodes")

# Plot 2: Larger educational code
plt.subplot(2, 3, 2)
H_edu2 = educational_codes["Educational (8,3)"]["parity_check_matrix"]
plt.imshow(H_edu2, cmap="RdYlBu_r", interpolation="nearest", aspect="auto")
for i in range(H_edu2.shape[0]):
    for j in range(H_edu2.shape[1]):
        plt.text(j, i, int(H_edu2[i, j]), ha="center", va="center", color="white" if H_edu2[i, j] == 1 else "black", fontweight="bold")
plt.title("Educational (8,3)\nParity Check Matrix", fontsize=11)
plt.xlabel("Variable Nodes")
plt.ylabel("Check Nodes")

# Plot 3: Professional code complexity illustration
plt.subplot(2, 3, 3)
if professional_codes:
    prof_code = list(professional_codes.values())[0]
    H_prof = prof_code["encoder"].check_matrix
    # Show only a small portion to illustrate complexity
    H_sample = H_prof[:20, :40].cpu()  # Sample view
    plt.imshow(H_sample, cmap="RdYlBu_r", interpolation="nearest", aspect="auto")
    plt.title(f"Professional Code Sample\n({prof_code['n']},{prof_code['k']}) - Partial View", fontsize=11)
    plt.xlabel("Variable Nodes (sample)")
    plt.ylabel("Check Nodes (sample)")
    plt.text(20, 10, f"Full size:\n{H_prof.shape[0]}×{H_prof.shape[1]}", bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Plot 4: Complexity comparison
plt.subplot(2, 3, 4)
code_types = ["Educational\n(6,3)", "Educational\n(8,3)"]
block_lengths = [6, 8]
info_bits = [3, 3]

if professional_codes:
    for name, code in professional_codes.items():
        code_types.append(f"Professional\n{name}")
        block_lengths.append(code["n"])
        info_bits.append(code["k"])

colors = ["lightblue", "lightgreen"] + ["salmon"] * len(professional_codes)
bars = plt.bar(code_types, block_lengths, color=colors, alpha=0.7)
plt.ylabel("Block Length (bits)")
plt.title("Block Length Comparison")
plt.xticks(rotation=45)

# Add value labels on bars
for bar, length in zip(bars, block_lengths):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(length), ha="center", va="bottom", fontweight="bold")

# Plot 5: Information content comparison
plt.subplot(2, 3, 5)
bars = plt.bar(code_types, info_bits, color=colors, alpha=0.7)
plt.ylabel("Information Bits per Block")
plt.title("Information Content Comparison")
plt.xticks(rotation=45)
plt.yscale("log")

# Add value labels
for bar, bits in zip(bars, info_bits):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1, str(bits), ha="center", va="bottom", fontweight="bold")

# Plot 6: Message space comparison
plt.subplot(2, 3, 6)
message_spaces = [2**k for k in info_bits]
bars = plt.bar(code_types, message_spaces, color=colors, alpha=0.7)
plt.ylabel("Total Possible Messages")
plt.title("Message Space Complexity")
plt.xticks(rotation=45)
plt.yscale("log")

# Add scientific notation labels
for bar, space in zip(bars, message_spaces):
    if space < 1e10:
        label = f"{space:.0e}"
    else:
        label = f"2^{int(np.log2(float(space)))}"
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5, label, ha="center", va="bottom", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.suptitle("Educational vs Professional LDPC Codes: Structural Differences", fontsize=14, y=1.02)
plt.show()

# %%
# Appropriate Use Case Analysis
# --------------------------------------
# Show what each type is good for

print("\n" + "=" * 70)
print("APPROPRIATE USE CASES AND EVALUATION METHODS")
print("=" * 70)


def demonstrate_educational_use_case():
    """Demonstrate appropriate use of educational codes."""

    print("\n📚 EDUCATIONAL CODES - APPROPRIATE ANALYSIS")
    print("-" * 50)

    # Use educational code for teaching concepts
    edu_code = educational_codes["Educational (6,3)"]
    H = edu_code["parity_check_matrix"]
    encoder = LDPCCodeEncoder(check_matrix=H)

    print("✓ Perfect for teaching LDPC fundamentals:")
    print(f"  • Matrix structure visible: {H.shape}")
    print("  • All connections traceable")
    print("  • Tanner graph drawable by hand")
    print("  • Fast simulation for classroom demos")

    # Demonstrate educational analysis
    print("\n🔍 Educational Analysis Example:")
    print("  • Each variable node degree:", [torch.sum(H[:, j]).item() for j in range(H.shape[1])])
    print("  • Each check node degree:", [torch.sum(H[i, :]).item() for i in range(H.shape[0])])
    print("  • Code is regular:", len(set(torch.sum(H, dim=0).tolist())) == 1)

    # Quick simulation for demonstration
    # Note: Creating components to show feasibility of exhaustive testing
    _ = AWGNChannel(snr_db=6.0)  # Channel for potential simulation
    _ = BeliefPropagationDecoder(encoder, bp_iters=10)  # Decoder for potential simulation

    # Test all possible messages (only 8 total!)
    all_messages = torch.tensor([[i // 4, (i // 2) % 2, i % 2] for i in range(8)], dtype=torch.float32)

    print(f"\n✓ Can test ALL possible messages (only {len(all_messages)}):")
    _ = encoder(all_messages)  # Encode all messages to verify feasibility
    print("  All messages encoded successfully")
    print("  Perfect for exhaustive analysis and verification")


def demonstrate_professional_use_case():
    """Demonstrate appropriate use of professional codes."""

    print("\n🏭 PROFESSIONAL CODES - APPROPRIATE ANALYSIS")
    print("-" * 50)

    if not professional_codes:
        print("No professional codes available for demonstration")
        return

    prof_code = list(professional_codes.values())[0]
    _ = prof_code["encoder"]  # Encoder available for simulation

    print("✓ Designed for real-world deployment:")
    print(f"  • Standards compliant: {prof_code.get('standard', 'Industry standard')}")
    print(f"  • Block length optimized: {prof_code['n']} bits")
    print(f"  • Rate optimized: {prof_code['rate']:.3f}")
    print("  • Field proven in millions of devices")

    print("\n🔍 Professional Analysis Focus:")
    print(f"  • Total message space: 2^{prof_code['k']} ≈ 10^{prof_code['k']*np.log10(2):.0f}")
    print("  • Requires statistical analysis (exhaustive testing impossible)")
    print("  • Performance near Shannon limit")
    print("  • Optimized for specific channel conditions")

    # Realistic simulation setup
    print("\n✓ Requires realistic simulation:")
    print("  • Sample-based testing (exhaustive impossible)")
    print("  • Multiple SNR points for characterization")
    print("  • Convergence analysis at realistic operating points")
    print("  • Implementation complexity considerations")


demonstrate_educational_use_case()
demonstrate_professional_use_case()

# %%
# Fair Evaluation Methodologies
# --------------------------------------
# Show how to properly evaluate each type

print("\n" + "=" * 70)
print("FAIR EVALUATION METHODOLOGIES")
print("=" * 70)


def fair_educational_evaluation():
    """Show proper evaluation methods for educational codes."""

    print("\n📊 EDUCATIONAL CODE EVALUATION")
    print("-" * 40)

    edu_code = educational_codes["Educational (6,3)"]
    H = edu_code["parity_check_matrix"]
    encoder = LDPCCodeEncoder(check_matrix=H)

    print("✓ Appropriate evaluation metrics:")
    print("  • Structural analysis (degree distribution, regularity)")
    print("  • Exhaustive error pattern testing")
    print("  • Step-by-step BP algorithm demonstration")
    print("  • Theoretical vs simulated performance")
    print("  • Complexity analysis (simple enough to hand-calculate)")

    # Demonstrate structural analysis
    n, k = edu_code["n"], edu_code["k"]
    rate = edu_code["rate"]

    print("\n📐 Structural Properties:")
    print(f"  • Code parameters: ({n},{k}) rate={rate}")
    print("  • Minimum distance (theoretical bounds)")
    print("  • Girth analysis (shortest cycle in Tanner graph)")

    # Demonstrate educational performance analysis
    print("\n🎯 Educational Performance Analysis:")
    snr_range = np.arange(0, 8, 1)
    decoder = BeliefPropagationDecoder(encoder, bp_iters=5)

    # Small scale simulation appropriate for education
    ber_values = []
    for snr_db in snr_range:
        channel = AWGNChannel(snr_db=snr_db)
        ber_metric = BitErrorRate()

        # Use small number appropriate for classroom
        messages = torch.randint(0, 2, (50, k), dtype=torch.float32)
        codewords = encoder(messages)
        bipolar_codewords = 1 - 2.0 * codewords
        received_soft = channel(bipolar_codewords)
        decoded_messages = decoder(received_soft)

        ber_metric.update(messages, decoded_messages)
        ber_values.append(ber_metric.compute().item())

    # Plot educational analysis
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(snr_range, ber_values, "bo-", linewidth=2, markersize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Educational Code: BER vs SNR\n(Small scale, appropriate for teaching)")

    plt.subplot(1, 2, 2)
    # Show Tanner graph concept (simplified)
    variable_nodes = range(n)
    check_nodes = range(H.shape[0])

    # Create a simple bipartite graph visualization
    for i, check_node in enumerate(check_nodes):
        plt.scatter([0], [i], c="red", s=100, label="Check nodes" if i == 0 else "")
        for j, variable_node in enumerate(variable_nodes):
            if H[i, j] == 1:
                plt.plot([0, 1], [i, j], "k-", alpha=0.5)

    for j, variable_node in enumerate(variable_nodes):
        plt.scatter([1], [j], c="blue", s=100, label="Variable nodes" if j == 0 else "")

    plt.xlim(-0.5, 1.5)
    plt.title("Tanner Graph Structure\n(Visualizable for education)")
    plt.xticks([0, 1], ["Check\nNodes", "Variable\nNodes"])
    plt.legend()

    plt.tight_layout()
    plt.show()


def fair_professional_evaluation():
    """Show proper evaluation methods for professional codes."""

    print("\n📊 PROFESSIONAL CODE EVALUATION")
    print("-" * 40)

    if not professional_codes:
        print("No professional codes available")
        return

    prof_code = list(professional_codes.values())[0]
    encoder = prof_code["encoder"]

    print("✓ Appropriate evaluation metrics:")
    print("  • Standards compliance testing")
    print("  • Statistical performance analysis")
    print("  • Implementation complexity metrics")
    print("  • Real channel condition testing")
    print("  • Throughput and latency analysis")

    print("\n🎯 Professional Evaluation Focus:")
    print("  • Large-scale statistical analysis")
    print("  • Multiple operating points")
    print("  • Realistic channel models")
    print("  • Implementation trade-offs")
    print("  • Standards verification")

    # Demonstrate appropriate professional analysis
    print("\n📈 Professional Performance Analysis:")
    k = prof_code["k"]  # Information bits

    # Appropriate scale for professional evaluation
    snr_range = np.arange(0, 8, 2)  # Fewer points, focus on key operating regions
    decoder = BeliefPropagationDecoder(encoder, bp_iters=10)

    timing_data = []
    ber_values = []

    for snr_db in snr_range:
        channel = AWGNChannel(snr_db=snr_db)
        ber_metric = BitErrorRate()

        # Professional scale simulation
        num_samples = 50  # Reduced for demo, real analysis would use thousands
        messages = torch.randint(0, 2, (num_samples, k), dtype=torch.float32)
        codewords = encoder(messages)
        bipolar_codewords = 1 - 2.0 * codewords
        received_soft = channel(bipolar_codewords)

        # Measure complexity
        start_time = time.time()
        decoded_messages = decoder(received_soft)
        decode_time = time.time() - start_time

        timing_data.append(decode_time * 1000)  # ms
        ber_metric.update(messages, decoded_messages)
        ber_values.append(ber_metric.compute().item())

    # Plot professional analysis
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(snr_range, ber_values, "rs-", linewidth=2, markersize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f'Professional Code: {prof_code["name"]}\nStatistical Performance Analysis')

    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(snr_range)), timing_data, alpha=0.7, color="orange")
    plt.xlabel("SNR Point")
    plt.ylabel("Decoding Time (ms)")
    plt.title("Implementation Complexity\n(Decoding Time Analysis)")
    plt.xticks(range(len(snr_range)), [f"{snr} dB" for snr in snr_range])

    # Add value labels
    for bar, time_val in zip(bars, timing_data):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{time_val:.1f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


fair_educational_evaluation()
fair_professional_evaluation()

# %%
# Key Insights and Recommendations
# --------------------------------

print("\n" + "=" * 70)
print("KEY INSIGHTS AND RECOMMENDATIONS")
print("=" * 70)

print("\n🎯 WHY DIRECT COMPARISON IS INAPPROPRIATE:")
print("-" * 45)
print("1. DIFFERENT PURPOSES:")
print("   • Educational codes: Teaching and understanding")
print("   • Professional codes: Real-world deployment")

print("\n2. DIFFERENT SCALES:")
print("   • Educational: 3-bit messages (8 total possibilities)")
print("   • Professional: 288-448 bit messages (astronomical possibilities)")

print("\n3. DIFFERENT EVALUATION CRITERIA:")
print("   • Educational: Clarity, simplicity, demonstrability")
print("   • Professional: Performance, efficiency, standards compliance")

print("\n4. DIFFERENT OPERATING REGIMES:")
print("   • Educational: Ideal for algorithm demonstration")
print("   • Professional: Optimized for specific real-world conditions")

print("\n✅ PROPER USAGE GUIDELINES:")
print("-" * 30)
print("Educational Codes - Use for:")
print("  ✓ Teaching LDPC fundamentals")
print("  ✓ Algorithm development and testing")
print("  ✓ Theoretical analysis and verification")
print("  ✓ Visualization and demonstration")
print("  ✓ Proof-of-concept implementations")

print("\nProfessional Codes - Use for:")
print("  ✓ Real system implementations")
print("  ✓ Standards compliance testing")
print("  ✓ Baseline performance comparisons")
print("  ✓ Research benchmarking")
print("  ✓ Production deployment")

print("\n🎓 EDUCATIONAL VALUE:")
print("-" * 20)
print("This comparison teaches us:")
print("• Importance of appropriate benchmarking")
print("• Different tools for different purposes")
print("• Evolution from academic concepts to real systems")
print("• Why professional standards exist and dominate")

print("\n" + "=" * 70)
print("CONCLUSION: Educational and professional LDPC codes serve")
print("complementary purposes. Compare them for their intended use,")
print("not against each other in direct performance metrics.")
print("=" * 70)

# %%
# Summary Comparison Table
# ------------------------


def create_summary_table():
    """Create a comprehensive comparison table."""

    plt.figure(figsize=(14, 8))

    # Create comparison data
    categories = ["Block Length", "Information Bits", "Message Space", "Primary Purpose", "Best Use Case", "Evaluation Method", "Simulation Scale", "Analysis Focus"]

    educational_data = ["6-8 bits", "3 bits", "8 messages", "Education", "Teaching concepts", "Exhaustive testing", "Small scale", "Algorithm understanding"]

    professional_data = ["576-672 bits", "288-448 bits", "10^87 messages", "Production", "Real deployment", "Statistical analysis", "Large scale", "Performance optimization"]

    # Create table
    table_data = []
    for i, cat in enumerate(categories):
        table_data.append([cat, educational_data[i], professional_data[i]])

    # Plot as table
    ax = plt.gca()
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(cellText=table_data, colLabels=["Aspect", "Educational Codes", "Professional Codes"], cellLoc="center", loc="center", colWidths=[0.25, 0.35, 0.35])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(categories) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(weight="bold", color="white")
            elif j == 0:  # Category column
                cell.set_facecolor("#E8F5E8")
                cell.set_text_props(weight="bold")
            elif j == 1:  # Educational column
                cell.set_facecolor("#E3F2FD")
            else:  # Professional column
                cell.set_facecolor("#FFF3E0")

    plt.title("Educational vs Professional LDPC Codes: Comprehensive Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.show()


create_summary_table()
