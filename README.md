This project analyzes academic collaboration patterns using raw data from the Open Academic Graph (OAG). It builds a coauthorship network from .tsv files, extracts edge-level features such as coauthorship frequency, citation impact, and publication year, and utilizes machine learning to classify strong vs. weak collaborations. 

## Setup Instructions

1. **Install Requirements**

   ***In AcademicCollabProject***, install all necessary Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Data**

   - Create a directory named `data` in the project root:

     ```bash
     mkdir data
     ```

   - Download and save the raw Open Academic Graph (OAG) files from this Google Drive folder: **(Make sure to extract zip, all raw data should be in same directory)**

     [OAG Data on Google Drive](https://drive.google.com/drive/folders/1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz)

   - Place all downloaded files into the newly created `data/` directory.

**In your visual studio solution, it should look something like this:**

![image](https://github.com/user-attachments/assets/5c44ee2b-a663-441b-ae91-486005a15da0)


3. **Run the Code**

   Open and execute the `main.ipynb` Jupyter notebook to run the full pipeline.
---
**Sources:**
Raw OAG data imported from:
[OAG Data Source](https://github.com/acbull/pyHGT/tree/master?tab=readme-ov-file)
