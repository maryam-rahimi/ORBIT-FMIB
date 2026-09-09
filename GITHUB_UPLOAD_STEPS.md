# Upload this package to GitHub

## 1. Create the repository

On GitHub choose **New repository** and use:

- Repository name: `ORBIT-FMIB`
- Description: `Order-resolved tracing of epistatic information through frozen ESM-2 representations on GB1.`
- Visibility during double-blind review: **Private** unless you are using a genuinely anonymous GitHub account/organization.
- Do not initialize with a README, `.gitignore`, or license; those files are already in this package.

## 2. Unzip locally

Unzip `ORBIT-FMIB_GitHub_Anonymous_Final.zip`. Open a terminal inside the extracted `ORBIT-FMIB` folder.

## 3. Initialize Git

```bash
git init
git branch -M main
git add .
git commit -m "Initial ORBIT-FMIB reproducibility release"
```

## 4. Connect the GitHub repository

```bash
git remote add origin https://github.com/<ACCOUNT>/ORBIT-FMIB.git
git push -u origin main
```

Replace `<ACCOUNT>` with the owner of the repository.

## 5. Double-blind warning

A repository hosted under a personal account is not anonymous even when the files are anonymized. During review, either keep the personal repository private and submit the anonymized ZIP as supplementary material, or use a genuinely anonymous repository/account with no identifying profile or Git history.

Do not add author names, institutional affiliations, personal email addresses, cluster usernames, or identifying commit metadata to the anonymous review copy.
