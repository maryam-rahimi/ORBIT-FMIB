# Upload this package to GitHub

## 1. Create the repository

On GitHub choose **New repository** and use:

- Repository name: `ORBIT-FMIB`
- Description: `Order-resolved tracing of epistatic information through frozen ESM-2 representations on GB1.`
- Visibility: **Public**. The target venue (RECOMB 2027) uses single-blind review and requires the source code to be publicly accessible, with a documented README, by the submission deadline.
- Do not initialize with a README, `.gitignore`, or license; those files are already in this package.

## 2. Unzip locally

Unzip the release package. Open a terminal inside the extracted `ORBIT-FMIB` folder.

## 3. Initialize Git

```bash
git init
git branch -M main
git add .
git commit -m "ORBIT-FMIB reproducibility release"
```

## 4. Connect the GitHub repository

```bash
git remote add origin https://github.com/<ACCOUNT>/ORBIT-FMIB.git
git push -u origin main
```

Replace `<ACCOUNT>` with the owner of the repository.

## 5. Tag the release

After pushing, create a GitHub release (for example `v2.0-recomb`) so the code cited in the paper is pinned to a fixed commit.

## 6. Check from outside

Open the repository in a private browser window and confirm that it loads without signing in, that `03 Code/` lists all 28 scripts, and that the README renders correctly.
