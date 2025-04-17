import os
import shutil
import subprocess  # nosec B404 # Add import for subprocess

# Change working directory to the root project directory
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

output_text = ""
# Update template file path
template_file_path = os.path.join("scripts", "README_template.rst")
readme_rst_path = "README.rst"
readme_md_path = "README.md"

try:
    with open(template_file_path) as template_file:
        for line in template_file:
            line = line.strip("\n").replace(".. include:: ", "")
            if line.endswith(".rst"):
                include_file_path = line
                try:
                    with open(include_file_path) as included_file:
                        included_text = included_file.read()

                        if include_file_path == "docs/license.rst":
                            included_text = "\n".join(included_text.split("\n")[:4])
                            included_text += "\n"

                        output_text += included_text.replace(":class:", " ")
                except FileNotFoundError:
                    print(f"Error: Included file '{include_file_path}' not found.")
                except Exception as e:
                    print(f"Error processing included file '{include_file_path}': {e}")
            else:
                output_text += line + "\n"

    print(output_text)

    # Write the generated RST content to README.rst
    with open(readme_rst_path, "w+") as readme_file:
        readme_file.write(output_text)

    # Convert README.rst to README.md using pandoc
    try:
        pandoc_path = shutil.which("pandoc")
        if not pandoc_path:
            print("Error: pandoc executable not found in PATH.")
            exit(1)

        print(f"Converting {readme_rst_path} to {readme_md_path} using pandoc...")
        subprocess.run(  # nosec B603 # subprocess call is necessary and arguments are controlled
            [pandoc_path, readme_rst_path, "-f", "rst", "-t", "markdown", "-o", readme_md_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Conversion successful.")

        # Post-process the Markdown file to fix image alignment
        with open(readme_md_path) as md_file:
            md_content = md_file.read()

        # Replace Markdown images with align-center attribute to HTML centered div
        import re

        md_content = re.sub(r"!\[(.*?)\]\((.*?)\)(\{\.align-center\})", r'<div align="center">\n<img src="\2" alt="\1">\n</div>', md_content)

        # Replace triple dashes with single dash in headings
        md_content = re.sub(r"(#\s+.*?) --- (.*?)", r"\1 - \2", md_content)

        # Remove all newlines between badges - this combines all badge lines into one
        badge_pattern = r"(\[\!\[.*?\]\(.*?\)\]\(.*?\))"
        md_content = re.sub(f"{badge_pattern}\n\n{badge_pattern}", r"\1 \2", md_content)

        # Run the replacement multiple times to handle groups of badges
        for _ in range(10):  # 10 iterations should be enough for all badges
            md_content = re.sub(f"{badge_pattern}\n\n{badge_pattern}", r"\1 \2", md_content)

        # Write the fixed content back to the README.md file
        with open(readme_md_path, "w") as md_file:
            md_file.write(md_content)

        print("Post-processed Markdown file to fix image alignment.")

        # Remove the intermediate README.rst file
        try:
            os.remove(readme_rst_path)
            print(f"Removed intermediate file: {readme_rst_path}")
        except OSError as e:
            print(f"Error removing file {readme_rst_path}: {e}")

    except FileNotFoundError:
        print("Error: pandoc command not found. Please ensure pandoc is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error during pandoc conversion: {e}")
        print(f"Pandoc stderr: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")


except FileNotFoundError:
    print(f"Error: Template file '{template_file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Copy the contents of docs/changelog.rst to CHANGES.txt
try:
    with open("docs/changelog.rst") as changelog_file:
        changelog_text = changelog_file.read()
        with open("CHANGES.txt", "w+") as changes_file:
            changes_file.write(changelog_text)
except FileNotFoundError:
    print("Error: Changelog file 'docs/changelog.rst' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Add Installation section after Features
try:
    simplified_installation = """# Installation

The fastest way to install Kaira is directly from PyPI:

```bash
pip install kaira
```
"""

    # Read the current README.md
    with open(readme_md_path) as md_file:
        md_content = md_file.read()

    # Remove any existing Installation sections first
    installation_pattern = r"#\s+(?:🔧\s+)?Installation[\s\S]+?(?=^#\s+[^#])"
    md_content = re.sub(installation_pattern, "", md_content, flags=re.MULTILINE)

    # Find the Features section
    features_pattern = r"(#\s+(?:✨\s+)?Features[\s\S]+?)(?=^#\s+[^#])"
    features_match = re.search(features_pattern, md_content, re.MULTILINE)

    if features_match:
        # Insert the installation section after the Features section
        md_content = md_content.replace(features_match.group(1), features_match.group(1) + "\n\n" + simplified_installation)

        # Write back to README.md
        with open(readme_md_path, "w") as md_file:
            md_file.write(md_content)

        print("Added Installation section after Features")
    else:
        print("Warning: Could not find Features section")

    # Fix the Citing Kaira section formatting if needed
    citing_pattern = r"\\#\s+Citing Kaira"
    citing_replacement = "# Citing Kaira"
    md_content = re.sub(citing_pattern, citing_replacement, md_content)

    # Fix any broken code blocks in the citing section
    broken_bibtex = r"``\s*`bibtex\s+@software\{.*?\}\s+``\\`"
    if re.search(broken_bibtex, md_content, re.DOTALL):
        fixed_bibtex = """```bibtex
@software{kaira2025,
  title = {Kaira: A {PyTorch}-based toolkit for simulating communication systems},
  author = {{Kaira Contributors}},
  year = {2025},
  url = {https://github.com/ipc-lab/kaira},
  version = {0.1.0}
}
```"""
        md_content = re.sub(broken_bibtex, fixed_bibtex, md_content, flags=re.DOTALL)

    # Write back to README.md
    with open(readme_md_path, "w") as md_file:
        md_file.write(md_content)

except Exception as e:
    print(f"Error replacing Installation section: {e}")
