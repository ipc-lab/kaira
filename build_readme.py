output_text = ""
template_file_path = "README_template.rst"

try:
    with open(template_file_path) as template_file:
        for line in template_file:
            line = line.strip("\n").replace(".. include:: ", "")
            if line.endswith(".rst"):
                include_file_path = line  # os.path.join(".", line) if you need relative path
                try:
                    with open(include_file_path) as included_file:
                        included_text = included_file.read()
                        
                        if include_file_path == "docs/license.rst":
                            included_text = included_text.split("\n")[0]
                            included_text += "\n"
                            
                        output_text += included_text.replace(":class:", " ")
                except FileNotFoundError:
                    print(f"Error: Included file '{include_file_path}' not found.")
                except Exception as e:
                    print(f"Error processing included file '{include_file_path}': {e}")
            else:
                output_text += line + "\n"

    print(output_text)

    with open("README.rst", "w+") as readme_file:
        readme_file.write(output_text)

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
