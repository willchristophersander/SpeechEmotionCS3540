# SSH Connection Troubleshooting for UVM Silk

## Issue: Connection Refused

If you're getting "Connection refused" when trying to SSH to Silk, try these solutions:

## Solution 1: Check Network Connection

**You need to be on UVM's network or VPN:**

1. **On UVM Campus**: Should work directly
2. **Off Campus**: Connect to UVM VPN first
   - UVM VPN information: https://www.uvm.edu/it/kb/uvm-vpn
   - Use your UVM credentials to connect

## Solution 2: Try Alternative Hostnames

Try these different hostnames:

```bash
# Option 1: w3.uvm.edu (standard)
ssh wsander@w3.uvm.edu

# Option 2: silk.uvm.edu (alternative)
ssh wsander@silk.uvm.edu

# Option 3: Full domain
ssh wsander@w3.uvm.edu

# Option 4: Using the SSH config alias
ssh uvm-silk
```

## Solution 3: Check SSH Port

Some servers use non-standard ports. Try:

```bash
ssh -p 22 wsander@w3.uvm.edu
```

## Solution 4: Use Cursor Remote SSH (Recommended)

Instead of using terminal SSH, use Cursor's built-in Remote SSH:

1. **In Cursor**: Press `Cmd+Shift+P` (or `Ctrl+Shift+P`)
2. Type: **"Remote-SSH: Connect to Host"**
3. Select: **"uvm-silk"** (from your SSH config)
4. Enter your password when prompted
5. Select platform: **Linux**
6. Open folder: **`~/www-root`** or **`~`**

This is often more reliable than terminal SSH.

## Solution 5: Check if Server is Up

The server might be temporarily down. Contact:
- **SAA (Systems Architecture & Administration)** at UVM
- Check UVM IT status page if available

## Solution 6: Verify Your Account

Make sure your Silk account is active:
- Email SAA to verify your account status
- Confirm your NetID (wsander) is correct

## Solution 7: Clear SSH Known Hosts (if needed)

If you've had connection issues before:

```bash
ssh-keygen -R w3.uvm.edu
ssh-keygen -R silk.uvm.edu
```

## Quick Test Commands

```bash
# Test connectivity
ping -c 3 w3.uvm.edu

# Test SSH with verbose output
ssh -v wsander@w3.uvm.edu

# Test with different hostname
ssh -v wsander@silk.uvm.edu
```

## Alternative: Upload Files First, Then Connect

If SSH isn't working, you could:

1. **Package files locally:**
   ```bash
   cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
   tar -czf silk_deploy.tar.gz --exclude='__pycache__' --exclude='*.pyc' .
   ```

2. **Upload via web interface** (if Silk has one)
   - Check if Silk has a web-based file manager
   - Or use SFTP client like FileZilla

3. **Once files are uploaded**, connect via Cursor Remote SSH to configure

## Contact UVM IT

If nothing works:
- **Email SAA**: Check UVM IT website for current contact
- **Phone**: UVM IT Help Desk
- Mention: "SSH connection refused to w3.uvm.edu for Silk web hosting"

## Most Likely Solution

**You probably need to connect to UVM VPN first** if you're off-campus.

Try connecting to UVM VPN, then:
```bash
ssh uvm-silk
# or
ssh wsander@w3.uvm.edu
```

