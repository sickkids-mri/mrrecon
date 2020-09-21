# Contributing Guidelines

Thank you for your interest in contributing to this repository and for reading these contributing guidelines.

Since this repository will be shared between all Python projects, it is important that all code is reviewed through a pull request. Do not commit your changes directly to the master branch! A pull request can be created by following the branching workflow or forking workflow. The branching workflow will be explained below.

## Branching Workflow

This repository follows a workflow that contains a master branch and feature branches, with the intention that all feature branches will eventually be merged into master once approved. The master branch should contain code that works and can be deployed. Users of the repository, who are not developing, should only need to look in the master branch.

The branching workflow will be outlined below.

First make a local copy of the repository on your workstation.
```
git clone https://github.com/sickkids-mri/mrrecon.git
cd mrrecon
```
If you've already cloned this repository previously, make sure the master branch is up to date before you start making changes.
```
git checkout master
git pull
```
Create a new branch off of the master branch and switch to it. While on the master branch,
```
git checkout -b name-of-new-feature
```
Now you can create new files or edit existing files. Group your changes appropriately and commit them to your new branch. The workflow here should look something like this:
```
## Make some changes ##

git add file_with_changes.py
git add another_file_with_changes.py
git commit -m "Add density compensation factor for centre out radial trajectory"

## Make some changes ##

git add file_with_changes.py
git commit -m "Improve documentation for density compensation factor"
```
It is good practice to make commit messages like this; generally one sentence long, the first letter in the sentence should be capitalized, and the one sentence should not end with a period.

Once the work is done, make sure that your master branch is still up to date since other developers may have merged new features to it in the time that you were working. If this is the case, you will also need to rebase your branch so that a simple fast-forward merge can be performed.
```
# Get new changes from other developers onto your master branch
git checkout master
git pull
# Rebase your branch
git checkout name-of-new-feature
git rebase master
```
Now your branch can be pushed to the remote repository. Since your new branch does not currently exist on the remote repository, you will have to indicate where to push to.
```
git push -u origin name-of-new-feature
```
Subsequent pushes can be made by just doing
```
git push
```
The new branch and its commits will now appear on GitHub and a pull request can be started by switching to the new branch on GitHub and pressing "Compare & pull request". Add a title and description to the pull request. Be sure to assign a reviewer to the pull request before pressing "Create pull request".

After review, changes to your branch may have to be made. Simply perform any changes locally, add, commit, rebase, and push and the pull request will automatically be updated.

After the pull request is approved and changes are merged, the feature branch can be deleted if no more work will be done on that branch. First delete the feature branch on GitHub by simply pressing "Delete branch" from within the pull request. Then, in the local repo, switch back to the master branch and pull the newly merged changes. Then delete the feature branch and prune any irrelevant references.
```
git checkout master
git pull
git branch -d name-of-new-feature
git fetch -p
```

## Coding Style

To keep code easy to read, Python code should follow the [PEP-8 style guide](https://www.python.org/dev/peps/pep-0008/). Python docstrings should follow the [Google style guide for docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). For automatically checking that your code follows PEP-8, install a linter like [pycodestyle](https://pypi.org/project/pycodestyle/) or [flake8](https://pypi.org/project/flake8/) and enable it in your code editor.
