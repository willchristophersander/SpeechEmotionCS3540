# SSH Authentication Notes for UVM Silk

## Important: Off-Campus Authentication

According to UVM documentation:

### Password Authentication (Off-Campus)
- **Requires Duo multifactor authentication** if you're off-campus
- This might be causing connection issues if Duo isn't properly configured

### SSH Key Authentication (Recommended)
- **More reliable** for off-campus connections
- **Doesn't require Duo** for each connection
- **Avoids rate limiting** from failed password attempts

## Setting Up SSH Keys

If you haven't already, consider setting up SSH keys:

1. **Generate SSH key** (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@uvm.edu"
   ```

2. **Copy public key to Silk**:
   ```bash
   ssh-copy-id wsander@w3.uvm.edu
   ```
   Or manually add `~/.ssh/id_ed25519.pub` to `~/.ssh/authorized_keys` on Silk

3. **UVM Instructions**: https://www.uvm.edu/it/kb/article/ssh-keys/

## Current Issue

The "Connection refused" is likely due to:
1. **Rate limiting** from too many failed attempts (wait 15-30 min)
2. **Duo authentication** issues if using password off-campus
3. **Network restrictions** on your current network

## Solution

**Best approach**: Wait 15-30 minutes, then try Cursor Remote SSH again. If it still fails, consider setting up SSH keys for more reliable connections.

