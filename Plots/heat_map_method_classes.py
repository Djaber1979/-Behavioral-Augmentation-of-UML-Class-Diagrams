import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constructing the DataFrame
import pandas as pd
import matplotlib.pyplot as plt

# Construct the DataFrame based on the provided annotation completeness percentages
data = {
    'Model': ['GPT-4o', 'GPT-3.5-o3', 'Claude 3.7', 'DeepSeek-R1', 'Gemini 2.5', 
              'Grok 3', 'Llama 4', 'Mistral 8×7B', 'Qwen 3'],
    'Full compliance': [12.3, 9.1, 14.0, 8.5, 10.8, 13.5, 5.2, 11.9, 9.8],
    'Action-only': [7.8, 5.5, 9.2, 4.3, 6.1, 8.0, 2.1, 7.5, 5.9],
    'UC-only': [15.4, 12.7, 10.3, 11.0, 13.9, 14.2, 8.7, 12.1, 11.8],
    'No annotation': [64.5, 72.7, 66.5, 76.2, 69.2, 64.3, 84.0, 68.5, 72.5]
}

df = pd.DataFrame(data).set_index('Model')

# Plot 1: Stacked bar chart
ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_ylabel('Percentage of Methods (%)')
ax.set_title('Annotation Completeness per LLM (Stacked Bar)')
ax.legend(title='Annotation Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot 2: Grouped bar chart for Full compliance vs No annotation
fig, ax2 = plt.subplots(figsize=(10, 4))
subset = df[['Full compliance', 'No annotation']]
subset.plot(kind='bar', ax=ax2)
ax2.set_ylabel('Percentage of Methods (%)')
ax2.set_title('Full Compliance vs No Annotation per LLM')
ax2.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
