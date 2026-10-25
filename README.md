# Hyperspectral Ice core processing and Analysis

This repository provides tools for processing hyperspectral ice core imagery, including **data retrieval, preprocessing, bubble detection and analysis, and spectral summations**. It supports both interactive exploration in Jupyter notebooks and automated batch processing for large datasets.


## Features
- Load and preprocess hyperspectral ice core scans (ENVI format)
- Bubble Analysis:
  * Adaptive thresholding and watershed-based bubble segmentation
  * Measurement of bubble size, elongation, orientation, and depth
  * Bubble count and statistics per cm and across entire core sections
- Spectral Sum Processing
- Automated visualizations and CSV exports for further analysis
- Batch processing scripts for high-throughput workflows


## Installation
Clone the repository and set up the environment:
```bash
git clone https://hsi.um.maine.edu/hsi/stat.git
cd stat
```

Create conda environment:
```bash
conda create -n hsi python=3.10
conda activate hsi
pip install -r requirements.txt
```

Dependencies include Spectral Python (SPy), Scipy, numpy, pandas, matplotlib, opencv-python, scikit-image, scikit-learn, astropy, and photoutils.



# Usage
These notebooks and scripts require csv datafiles containing the Data-ID, Core name and section, image file path, and image metadata. Running a particular image requires referencing the DataID.
The file format:
```bash
...{DATA-DIR}/{DataDir}/{ScanDir} or
/data/icecores/July2024/ALHIC2302-30_0002_HSI_UM_Jul24
```
is implied in the code. The required csv file should be in the DATA-DIR. The code to run these images in not required to be in this DIR, it can be anywhere on the machine.

1. Interactive (Notebooks)
For exploration, testing, and single-scan workflows
```bash
jupyter lab
```
Open a notebook from the Notebooks/ folder to run step-by-step workflows.


2. Batch Processing
For processing multiple scans automatically
```bash
python spectralBubbles.py {DataID}
```
or
```bash
./runBubbles.py
```

Both the notebooks and scripts will require inputting DATA-DIR and OUTPUT-DIR


# Outputs
Follows file format:
```bash
...{OUTPUT-DIR}/{Region}/{Project}/{CoreName}/{CoreName-Section}/{TimeStamp}
```
For example: 
```bash
/data/icecores/Output/AN/ALHIC/ALHIC2302/ALHIC2302-29/2025-07-30_214105
```

Note: Folders after OUTPUT-DIR are auto generated when neccessary in running code

- Bubble Analysis Output
  * Core/Image Metadata
  * rgb of trimmed and filtered hsi
  * Bubble Mask
  * Bubble channel map
  * Bubble Analysis csv
  * Bubble Analysis per cm csv
  * Bubble stats csv
  * Bubble stats graph png
  * README.md

# Citation
If used in publications or academic work:
```bash
Rumsey, R. (2025). Hyperspectral Ice Core Processing and Analysis [Software].
```

---

Maintained by **Roisin Rumsey**



# Change Log

### spectralBubbles V 1.1 - 3/2025
- Added stats for bubbles
- Added dynamic output path




# To Use server jetstream server

- HostName: jetstream.um.maine.edu  
1. VPN to UMaine c
2. Login via ssh.  Open terminal on a local computer (e.g., macbook) and type: ssh jetstream.um.maine.edu
3. activate environmnt for python HSI work
4. Start A Jupyter Lab in headfless mode on port 8080 on the server by typing into terminal promt:
 
- jupyter-lab --no-browser --port=8080
- Note output after: Or copy and paste one of these URLs: e.g., http://localhost:8080/lab?token=5f66ee77a2eec4754be2c3db52a0d23985d886aa90a51b30
- Start a new Terminal window on a local computer (e.g., macbook)

1. Set up a SSH tunnel to the remote machine using the following command: 

- ssh -N -L 8080:localhost:8080 jetstream.um.maine.edu
- Open a web browser and paste output from the first terminal window into the url field:
-  http://localhost:8080/lab?token=xxxx

# ML links

- https://drive.google.com/file/d/1TX-6y__I9re1H38eRTjJFtpIiOzivyvm/view?usp=drive_link 
- https://www.fast.ai/
- https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&ab_channel=StanfordUniversitySchoolofEngineering


<!-- ## Clone the project
to clone the project
```
git clone git@hsi.um.maine.edu:hsi/hsi-tools.git
```
## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://hsi.um.maine.edu/hsi/stat.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://hsi.um.maine.edu/hsi/stat/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers. -->
