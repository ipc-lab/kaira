import os
import subprocess  # Add import for subprocess

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
        print(f"Converting {readme_rst_path} to {readme_md_path} using pandoc...")
        subprocess.run(
            ["pandoc", readme_rst_path, "-f", "rst", "-t", "markdown", "-o", readme_md_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Conversion successful.")

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
