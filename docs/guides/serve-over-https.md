# Serve over https

The server speaks plain http, which is fine on your own network for boards and scripts. Browsers, though, allow the microphone only at `localhost` or over https, and keys sent over plain http can be read by anyone on the network. Any of these gives you https:

- **Tailscale, as a proxy:** `tailscale serve --bg 5000` serves the server at `https://<machine>.<tailnet>.ts.net` to your tailnet, with a real certificate. Requests then come from this PC, so serve with `--behind-proxy`.
- **Tailscale, as a certificate:** `tailscale cert <machine>.<tailnet>.ts.net` writes the certificate and key, and `serve --certfile <machine>.<tailnet>.ts.net.crt --keyfile <machine>.<tailnet>.ts.net.key` serves https itself. The certificate lasts 90 days; run `tailscale cert` again before then.
- **mkcert, on your network:** `mkcert -install` makes a certificate authority on this PC, and `mkcert pc.lan 192.168.1.10` a certificate for the PC's names, as `pc.lan+1.pem` and `pc.lan+1-key.pem`, for `--certfile` and `--keyfile`. Other devices trust it once they install the authority's `rootCA.pem`, from the folder `mkcert -CAROOT` names.
- **Caddy, as a proxy:** with a Caddyfile like the one below, Caddy serves https and forwards to the server; serve with `--behind-proxy`. For a public name, leave out `tls internal` and Caddy gets a real certificate; for a local one, devices must trust Caddy's own authority.

```bash
lite-assistant serve --certfile pc.lan+1.pem --keyfile pc.lan+1-key.pem   # https://pc.lan:5000
```

```
pc.lan {
    tls internal
    reverse_proxy localhost:5000
}
```

With https, live mode's WebSocket is `wss://` too; the chat page picks that by itself.
