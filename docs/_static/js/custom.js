document.addEventListener('DOMContentLoaded', function() {
  // Add a progress bar for long pages
  const progressContainer = document.createElement('div');
  progressContainer.className = 'progress-container';
  const progressBar = document.createElement('div');
  progressBar.className = 'progress-bar';
  progressContainer.appendChild(progressBar);
  document.body.prepend(progressContainer);

  // Update the progress bar as user scrolls
  window.addEventListener('scroll', function() {
    const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    progressBar.style.width = scrolled + '%';
  });

  // Make tables responsive
  const tables = document.querySelectorAll('table');
  tables.forEach(table => {
    const wrapper = document.createElement('div');
    wrapper.style.overflowX = 'auto';
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });
  
  // Add copy button to code blocks
  const codeBlocks = document.querySelectorAll('div.highlight pre');
  codeBlocks.forEach(block => {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'Copy';
    button.style.position = 'absolute';
    button.style.top = '0.5rem';
    button.style.right = '0.5rem';
    button.style.padding = '0.25rem 0.5rem';
    button.style.fontSize = '0.8rem';
    button.style.border = 'none';
    button.style.borderRadius = '3px';
    button.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    button.style.color = '#e6f1ff';
    button.style.cursor = 'pointer';
    
    // Ensure the parent has a relative position for absolute positioning
    const parent = block.parentElement;
    parent.style.position = 'relative';
    
    button.addEventListener('click', () => {
      const code = block.textContent;
      navigator.clipboard.writeText(code).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => {
          button.textContent = 'Copy';
        }, 2000);
      });
    });
    
    parent.appendChild(button);
  });
  
  // Add smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      
      const target = document.querySelector(targetId);
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
        
        // Update URL without page jump
        history.pushState(null, null, targetId);
      }
    });
  });
});
