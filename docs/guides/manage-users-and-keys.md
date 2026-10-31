# Manage users and keys

A request from this PC needs no key: it comes from the **owner**, an admin made on first run and named after your account. Any other request needs a user's key, sent as OpenAI's are, `Authorization: Bearer <key>`. So until you add a user, only this PC can use the server, even though it listens on every address.

```bash
lite-assistant users add alice           # Prints alice's key, once
lite-assistant users add kitchen-board   # A board is a user too, with its own key
lite-assistant users                     # Lists them
lite-assistant users key alice           # A new key; the old one stops working
lite-assistant users admin alice         # Makes alice an admin; --off undoes it
lite-assistant users remove alice
```

- **Each user** has their own chats and stored responses, and can't see anyone else's.
- **Admins** also add and remove models. The owner is always an admin.
- **Keys** are kept only as hashes, in `users.json` in the data directory, readable only by the account the server runs as. Changes from the command line work without restarting the server.
- **Wrong keys:** after 10 from one address in a minute, that address waits a minute, even with the right key.
- **From this PC**, a key that isn't anyone's, such as the openai client's placeholder, still means the owner, and a user's key means that user.
- **Private addresses only:** a request from a public address is refused, even with a key. Only this PC, the local network (10.x, 172.16–31.x, 192.168.x, IPv6 `fd…` and `fe80…`) and Tailscale (100.64–127.x) get in, so forwarding the port from a router, or a global IPv6 address, doesn't put the server on the internet. On a server in the cloud, `serve --allow-public` lets public addresses in, with a key.
- **`--behind-proxy`:** behind a proxy on the same PC, such as Caddy, every request arrives from this PC, so `serve --behind-proxy` makes every request need a key.
- **The chat page** asks for a key when the server wants one, and keeps it in the browser. Settings shows who you are, and **Use another key** changes it. Live mode sends the key as a WebSocket subprotocol, `openai-insecure-api-key.<key>`, as OpenAI's Realtime API takes it from a browser.
