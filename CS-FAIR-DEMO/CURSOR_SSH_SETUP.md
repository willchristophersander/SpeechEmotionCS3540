# Connecting to UVM Silk via Cursor Remote SSH

## SSH Config Setup

I've added the UVM Silk server to your SSH config file at `~/.ssh/config`:

```
Host uvm-silk
  HostName w3.uvm.edu
  User wsander
  ForwardAgent yes
```

## Steps to Connect in Cursor

### 1. Open Remote Connection

1. Press `Cmd+Shift+P` (or `Ctrl+Shift+P` on Windows/Linux) to open the command palette
2. Type: **"Remote-SSH: Connect to Host"**
3. Select: **"uvm-silk"** (or type it if it doesn't appear)

### 2. Select Platform

Cursor will ask you to select the platform:
- Choose **"Linux"** (Silk runs on Linux servers)

### 3. Enter Password

When prompted, enter your UVM password (the one you use for SSH access)

### 4. Open Folder

Once connected, you'll be asked to open a folder:
- Navigate to: **`~/www-root`** (this is where your website files go)
- Or: **`~`** (home directory) if you want to see everything first

## Alternative: Connect via Command Palette

You can also:
1. Click the green "><" icon in the bottom-left corner of Cursor
2. Select "Connect to Host..."
3. Choose "uvm-silk"

## After Connecting

Once connected, you can:
- Browse files on Silk
- Edit files directly
- Open a terminal in Cursor (Terminal → New Terminal)
- Upload files using the file explorer

## Uploading Your Project

### Option 1: Using Cursor's File Explorer
1. Connect to uvm-silk via Remote SSH
2. Open `~/www-root` folder
3. Drag and drop files from your local machine, or
4. Use the terminal to `scp` files

### Option 2: Using Terminal in Cursor
Once connected, open a terminal in Cursor and run:
```bash
# From your local machine (in a new terminal, not the remote one)
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
scp -r * wsander@w3.uvm.edu:~/www-root/
```

### Option 3: Direct Copy in Remote Session
1. Connect to uvm-silk
2. Open terminal in Cursor (remote terminal)
3. Use `wget` or `curl` if you have the files hosted somewhere, or
4. Create files directly in Cursor's editor

## Troubleshooting

**"Host key verification failed"**
- Run: `ssh-keyscan w3.uvm.edu >> ~/.ssh/known_hosts`

**"Permission denied"**
- Make sure you're using the correct password
- Check if you need to set up SSH keys with UVM

**"Connection timeout"**
- Verify you're on UVM network or VPN
- Check if `w3.uvm.edu` is the correct hostname

**Can't see files after connecting**
- Make sure you opened the correct folder (`~/www-root`)
- Check file permissions: `ls -la ~/www-root`

## Quick Test

After connecting, test the connection:
```bash
pwd  # Should show your home directory
ls -la  # List files
cd www-root  # Navigate to web directory
```

## Next Steps After Upload

1. Install dependencies: `pip install -r requirements.txt`
2. Test the app: `python3 app.py` (or configure WSGI)
3. Contact SAA if you need help with web server configuration

