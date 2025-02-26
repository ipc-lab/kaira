from typing import Any, Callable, List, Optional
from kaira.core import BasePipeline

class SequentialPipeline(BasePipeline):
    """A pipeline that processes steps sequentially.
    
    Each step receives the output of the previous step as its input.
    """
    
    def __init__(self, steps: Optional[List[Callable]] = None):
        """Initialize the sequential pipeline.
        
        Args:
            steps: Optional initial list of processing steps
        """
        super().__init__()
        if steps:
            self.steps = list(steps)
        
    def add_step(self, step: Callable):
        """Add a processing step to the pipeline.
        
        Args:
            step: A callable function or object that processes input data
            
        Returns:
            The pipeline instance for method chaining
        """
        if not callable(step):
            raise TypeError("Step must be callable")
        self.steps.append(step)
        return self
        
    def remove_step(self, index: int):
        """Remove a processing step from the pipeline.
        
        Args:
            index: The index of the step to remove
            
        Returns:
            The pipeline instance for method chaining
            
        Raises:
            IndexError: If the index is out of range
        """
        if not 0 <= index < len(self.steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self.steps)-1})")
        self.steps.pop(index)
        return self
        
    def forward(self, input_data: Any) -> Any:
        """Execute the pipeline sequentially on the input data.
        
        Args:
            input_data: The initial data to process
            
        Returns:
            The final result after passing through all steps
        """
        result = input_data
        for step in self.steps:
            result = step(result)
        return result
