# SSH Access Guide - Xavier NX (Ethernet Configuration)

## Objective
Establish stable SSH and VS Code Remote-SSH connection to Xavier NX for development using wired Ethernet through local router.

---

## 1. Physical Setup
1. Connect Xavier NX to same router or switch as laptop using Ethernet cable
2. Keep laptop on Wi-Fi or Ethernet - both must be on same LAN
3. Power on Xavier and allow boot to desktop

---

## 2. Find Xavier's IP Address
On Xavier terminal (keyboard/monitor attached):

```bash
hostname -I
```

Look for IP such as 192.168.x.x or 10.x.x.x
Example: 192.168.1.102

If no address appears:

1. Open Settings → Network → Wired
2. Set IPv4 to Automatic (DHCP)
3. Toggle wired connection Off → On
4. Run `hostname -I` again

---

## 3. Verify Network Reachability

From laptop:

```bash
ping 192.168.1.102
```

Expected: Replies with latency <10 ms
If unreachable: Verify both devices on same router

---

## 4. SSH from Laptop (VS Code Microsoft Extension)

In VS Code:

1. Install extensions:
   - Remote - SSH
   - Remote - SSH: Editing Configuration Files

2. Edit SSH config file (~/.ssh/config):

```
Host xavier
    HostName 192.168.1.102
    User colin
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

3. In Command Palette, select:
   Remote-SSH: Connect to Host → xavier

---

## 5. Confirm Connection

When connected:

```bash
whoami
hostname
nvcc --version
```

Expected output:

```
colin
xavier
Cuda compilation tools, release 11.4
```

---

## 6. Tips

**IP Address Changes:**
- If SSH drops, re-check cable and IP (DHCP may assign new addresses)
- For permanent address, reserve Xavier's MAC in router

**Long Sessions:**
Use persistent shell for long builds:

```bash
sudo apt install tmux
tmux new -s build
```

Detach with `Ctrl+B D`
Reattach with `tmux attach -t build`

**Network Stability:**
- Wi-Fi on Xavier is not reliable for long GPU or compile sessions
- Use Ethernet for stable performance

---

## Summary

1. Use Ethernet + DHCP
2. Confirm LAN reachability
3. Connect through Microsoft Remote-SSH extension with keep-alive settings
4. Use tmux for long-running tasks

---

## Troubleshooting

**Cannot connect:**
- Verify IP with `hostname -I` on Xavier
- Check both devices on same network
- Verify SSH service running: `sudo systemctl status ssh`

**Connection drops:**
- Check Ethernet cable connection
- Verify router stability
- Consider static IP reservation

**Slow performance:**
- Avoid Wi-Fi, use Ethernet only
- Check network bandwidth with `iperf3`
- Monitor Xavier resources with `tegrastats`

---

## Quick Connect (denhac Network)

```bash
ssh colin@10.11.3.65
# Password: denhac rules
```

Or non-interactively:
```bash
sshpass -p 'denhac rules' ssh colin@10.11.3.65
```

**Verified:** 2026-01-26 — hostname `denhac`, kernel 5.10.104-tegra (aarch64)

**How this IP was found:** The address `10.11.3.65` is documented in
`envs/linux/README.md` as the Xavier's IP on the denhac network. It was
confirmed reachable via `ping` and authenticated via `sshpass` with the
credentials above.

---

**Platform:** NVIDIA Jetson Xavier NX
**JetPack:** 5.0.2 (L4T R35.1.0)
**Network:** Wired Ethernet recommended
**Last Updated:** 2026-01-26
