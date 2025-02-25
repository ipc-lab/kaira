Building Documentation
======================

To build the Kaira documentation, you need to use Sphinx, a powerful tool for creating and managing project documentation. Sphinx processes reStructuredText files to generate HTML, PDF, and other output formats.

Follow these steps to build and review the Kaira documentation using Sphinx: 

1. **Open a Terminal:**
   Launch the integrated terminal in Visual Studio Code.

2. **Navigate to the Documentation Folder:**
   Switch to the `docs` directory by running:

   .. code-block:: bash

      cd docs

3. **Build the HTML Documentation:**

   - **On Linux/macOS:**  
     Run the following command:

     .. code-block:: bash

        make html

   - **On Windows:**  
     Run the following command:

     .. code-block:: bat

        make.bat html

4. **View the Documentation:**
   Once the build completes, open the generated HTML files (located in the `_build/html` directory) in your browser.

Additional Tips
---------------
- Ensure you have all necessary dependencies installed (e.g., Sphinx and its extensions).
- You can re-build the documentation anytime after making changes to verify updates.
- For troubleshooting build issues, check the terminal output for error messages.

By following these steps, you'll have a local copy of the project documentation ready for review.
