# Google Drive Data Access Setup

This project reads data from a shared Google Drive folder. Each team member authenticates individually - **no shared credentials needed**.

## First-Time Setup (Each Person - One Time)

1. **Get OAuth Client ID:**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project (or select existing)
   - Enable **Google Drive API** (APIs & Services → API Library)
   - Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
   - Choose **Desktop application**
   - Download the JSON file

2. **Save to Repo:**
   - Rename the JSON file to `credentials.json`
   - Place it in the notebook folder
   - **Important:** Do NOT commit this file to Git (already in `.gitignore`)

3. **Run the Notebook:**
   - Execute the cell with `read_csvs_from_gdrive_folder()`
   - Browser opens → Log in with your Google account
   - Grants access to Drive → Closes automatically
   - `token.json` is saved locally (also in `.gitignore`)
   - **Subsequent runs:** Uses the saved token, no re-login needed

## How It Works

- **credentials.json**: OAuth2 credentials (personal, not shared)
- **token.json**: Access token generated after login (personal, not shared)
- **Both are in `.gitignore`** - each person has their own copy

## Team Access

All 3 team members are invited to the shared Drive folder. Each person:
1. Creates their own `credentials.json` from Google Cloud Console
2. Runs notebook → Authenticates once → Done
3. Can now access shared folder with their own `token.json`

## Troubleshooting

**"No credentials.json found"** → Create it following steps above

**"OAuth error"** → Make sure Google Drive API is enabled in Google Cloud Console

**"Permission denied"** → Confirm you're logged into correct Google account and folder is shared with you
