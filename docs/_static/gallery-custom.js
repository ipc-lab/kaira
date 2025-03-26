/**
 * Custom JavaScript enhancements for Sphinx-Gallery
 */
document.addEventListener('DOMContentLoaded', function() {
  // Add fade-in animation to gallery items
  const galleryItems = document.querySelectorAll('.sphx-glr-thumbcontainer');

  if (galleryItems.length > 0) {
    // Apply staggered animations to gallery thumbnails
    galleryItems.forEach((item, index) => {
      item.style.opacity = '0';
      item.style.transform = 'translateY(20px)';
      item.style.transition = 'opacity 0.5s ease, transform 0.5s ease';

      setTimeout(() => {
        item.style.opacity = '1';
        item.style.transform = 'translateY(0)';
      }, 100 + (index * 50)); // Staggered timing
    });

    // Add category filter if we have multiple sections
    const sections = document.querySelectorAll('.sphx-glr-section-title');
    if (sections.length > 1) {
      createCategoryFilter(sections);
    }

    // Search functionality removed
  }

  // Enhance syntax highlighting
  highlightActiveCodeLines();

  // Add scroll progress indicator for long examples
  addScrollProgress();

});

/**
 * Creates a category filter for the gallery
 * @param {NodeList} sections - The section titles
 */
function createCategoryFilter(sections) {
  // Create filter container
  const filterContainer = document.createElement('div');
  filterContainer.className = 'gallery-filter';
  filterContainer.innerHTML = `
    <div class="filter-heading">Filter by category:</div>
    <div class="filter-options"></div>
  `;

  // Style the container
  filterContainer.style.margin = '1rem 0 2rem 0';
  filterContainer.style.padding = '1rem';
  filterContainer.style.backgroundColor = '#f8fafc';
  filterContainer.style.borderRadius = '6px';
  filterContainer.style.boxShadow = '0 2px 4px rgba(0,0,0,0.05)';

  const filterOptions = filterContainer.querySelector('.filter-options');

  // Add "All" option
  const allOption = document.createElement('button');
  allOption.textContent = 'All';
  allOption.className = 'filter-btn active';
  allOption.dataset.category = 'all';
  filterOptions.appendChild(allOption);

  // Add option for each section
  sections.forEach(section => {
    const categoryName = section.textContent.trim();
    const categoryId = categoryName.toLowerCase().replace(/\s+/g, '-');

    const button = document.createElement('button');
    button.textContent = categoryName;
    button.className = 'filter-btn';
    button.dataset.category = categoryId;

    // Add section ID to its container
    const sectionContainer = section.parentElement;
    if (sectionContainer) {
      sectionContainer.id = categoryId;
    }

    filterOptions.appendChild(button);
  });

  // Style the buttons
  const buttons = filterOptions.querySelectorAll('.filter-btn');
  buttons.forEach(btn => {
    btn.style.padding = '0.5rem 1rem';
    btn.style.margin = '0.25rem';
    btn.style.border = '1px solid #e5e7eb';
    btn.style.borderRadius = '4px';
    btn.style.backgroundColor = '#ffffff';
    btn.style.cursor = 'pointer';
    btn.style.transition = 'all 0.2s ease';

    btn.addEventListener('mouseover', () => {
      if (!btn.classList.contains('active')) {
        btn.style.backgroundColor = '#f3f4f6';
      }
    });

    btn.addEventListener('mouseout', () => {
      if (!btn.classList.contains('active')) {
        btn.style.backgroundColor = '#ffffff';
      }
    });

    // Add click event
    btn.addEventListener('click', () => {
      // Remove active class from all buttons
      buttons.forEach(button => {
        button.classList.remove('active');
        button.style.backgroundColor = '#ffffff';
        button.style.borderColor = '#e5e7eb';
      });

      // Add active class to clicked button
      btn.classList.add('active');
      btn.style.backgroundColor = '#2563eb';
      btn.style.color = 'white';
      btn.style.borderColor = '#2563eb';

      // Filter sections
      const category = btn.dataset.category;
      sections.forEach(section => {
        const sectionContainer = section.parentElement;
        if (!sectionContainer) return;

        if (category === 'all') {
          sectionContainer.style.display = 'block';
        } else {
          if (sectionContainer.id === category) {
            sectionContainer.style.display = 'block';
          } else {
            sectionContainer.style.display = 'none';
          }
        }
      });
    });
  });

  // Add the filter before the first section
  if (sections.length > 0) {
    const firstSection = sections[0].parentElement;
    if (firstSection && firstSection.parentElement) {
      firstSection.parentElement.insertBefore(filterContainer, firstSection);
    }
  }
}

/**
 * Highlights the active line in code blocks on hover
 */
function highlightActiveCodeLines() {
  const codeBlocks = document.querySelectorAll('div.highlight pre');

  codeBlocks.forEach(block => {
    // Split code into lines and wrap each in a span
    const code = block.innerHTML;
    const lines = code.split('\n');
    let wrappedCode = '';

    // Process each line to remove extra newlines
    lines.forEach((line, index) => {
      // Remove empty lines that appear after non-empty lines
      if (line.trim() === '' && index > 0 && lines[index - 1].trim() !== '') {
        return;
      }

      // Add the line with proper wrapping
      if (line.trim() !== '') {
        wrappedCode += `<span class="code-line" data-line="${index + 1}">${line}</span>`;
      } else {
        wrappedCode += `<span class="code-line empty" data-line="${index + 1}">${line}</span>`;
      }
    });

    block.innerHTML = wrappedCode;

    // Style code lines
    const codeLines = block.querySelectorAll('.code-line');
    codeLines.forEach(line => {
      line.style.display = 'block';
      line.style.whiteSpace = 'pre';
      line.style.lineHeight = '1.4';
      line.style.margin = '0';
      line.style.padding = '0';

      // Add hover effect
      line.addEventListener('mouseover', () => {
        line.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
      });

      line.addEventListener('mouseout', () => {
        line.style.backgroundColor = 'transparent';
      });
    });
  });
}

/**
 * Adds a scroll progress indicator for long examples
 */
function addScrollProgress() {
  // Only add to example pages (not the gallery index)
  if (!document.querySelector('.sphx-glr-download-link-note')) {
    return;
  }

  const progressContainer = document.createElement('div');
  progressContainer.className = 'progress-container';
  progressContainer.innerHTML = '<div class="progress-bar"></div>';

  document.body.appendChild(progressContainer);

  const progressBar = progressContainer.querySelector('.progress-bar');

  // Update progress bar on scroll
  window.addEventListener('scroll', () => {
    const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    progressBar.style.width = scrolled + '%';
  });
}

/**
 * Adds copy buttons to code blocks
 */
function addCopyButtons() {
  const codeBlocks = document.querySelectorAll('div.highlight pre');

  codeBlocks.forEach(block => {
    // Make sure the block exists and has a parent element
    if (!block || !block.parentElement) return;

    // Create copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'copy-btn';
    copyButton.innerHTML = 'Copy';
    copyButton.style.position = 'absolute';
    copyButton.style.top = '5px';
    copyButton.style.right = '5px';
    copyButton.style.padding = '0.25rem 0.5rem';
    copyButton.style.fontSize = '0.8rem';
    copyButton.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    copyButton.style.border = 'none';
    copyButton.style.borderRadius = '4px';
    copyButton.style.color = '#e6f1ff';
    copyButton.style.cursor = 'pointer';
    copyButton.style.opacity = '0.6';
    copyButton.style.transition = 'opacity 0.2s ease, background-color 0.2s ease';

    // Make sure the block has position relative for absolute positioning
    block.parentElement.style.position = 'relative';

    // Add button to block
    block.parentElement.appendChild(copyButton);

    // Show on hover
    block.parentElement.addEventListener('mouseover', () => {
      copyButton.style.opacity = '1';
    });

    block.parentElement.addEventListener('mouseout', () => {
      copyButton.style.opacity = '0.6';
    });

    // Copy functionality
    copyButton.addEventListener('click', () => {
      const code = block.innerText.replace(/\n$/, '');

      try {
        navigator.clipboard.writeText(code).then(
          () => {
            copyButton.innerHTML = 'Copied!';
            copyButton.style.backgroundColor = '#10b981';

            setTimeout(() => {
              copyButton.innerHTML = 'Copy';
              copyButton.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
            }, 2000);
          },
          () => {
            copyButton.innerHTML = 'Failed';
            copyButton.style.backgroundColor = '#ef4444';

            setTimeout(() => {
              copyButton.innerHTML = 'Copy';
              copyButton.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
            }, 2000);
          }
        );
      } catch (err) {
        // Fallback for browsers that don't support clipboard API
        const textarea = document.createElement('textarea');
        textarea.value = code;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.select();

        try {
          document.execCommand('copy');
          copyButton.innerHTML = 'Copied!';
          copyButton.style.backgroundColor = '#10b981';
        } catch (err) {
          copyButton.innerHTML = 'Failed';
          copyButton.style.backgroundColor = '#ef4444';
        }

        document.body.removeChild(textarea);

        setTimeout(() => {
          copyButton.innerHTML = 'Copy';
          copyButton.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        }, 2000);
      }
    });
  });
}
