MVP architecture:

Users (Browser / Mobile)
        |
        v
   [OVHcloud DNS]
        |
        v
[OVHcloud VPS / Public Cloud Instance]
+---------------------------------------------------+
| NGINX (TLS via Let's Encrypt)                      |
|  - / -> web                                        |
|  - client_max_body_size                            |
+-------------------------+-------------------------+
                          |
                          v
                 +------------------+
                 | Flask + Gunicorn |
                 | (web container)  |
                 | OCR + DeepL      |
                 | uploads -> disk  |
                 +------------------+
                          |
          +---------------+----------------+
          |                                |
          v                                v
+-------------------+             +----------------------+
| Postgres (db)     |             | Prometheus /metrics  |
| (container)       |             | (optional container) |
+-------------------+             +----------------------+

Storage NOW:
- Images: local disk (./static/uploads)
- Cleanup: cron job deleting old files (e.g. 7–14 days)


Scale up on OVHcloud, still lean but ready for scaling:

Users
  |
  v
[OVHcloud DNS]
  |
  v
[OVHcloud Load Balancer (optional but recommended later)]
  |
  v
[OVHcloud Managed Kubernetes (MKS)  OR  your own K8s on Public Cloud]
+-------------------------------------------------------------------+
| Ingress (NGINX Ingress or Traefik)                                |
|                                                                   |
|  +------------------+        +------------------+                 |
|  | Web API          |        | Web Frontend     | (optional)      |
|  | (Flask/FastAPI)  |        | (templates/UI)   |                 |
|  +--------+---------+        +--------+---------+                 |
|           |                           |                           |
|           | enqueue job                \--> read status/results    |
|           v                                                       |
|   +------------------+                                           |
|   | Queue (Redis)    |  (OVHcloud Managed Redis or self-hosted)   |
|   +--------+---------+                                           |
|            |                                                     |
|            v                                                     |
|   +------------------+      +------------------+                 |
|   | Worker OCR       |      | Worker Translate |                 |
|   | (scale N pods)   |      | (scale N pods)   |                 |
|   +--------+---------+      +--------+---------+                 |
+------------|-------------------------|----------------------------+
             |                         |
             v                         v
   +----------------------+    +------------------------------+
   | OVHcloud Object      |    | OVHcloud Managed Postgres    |
   | Storage (S3-compatible)   | (or managed DB service)      |
   | - store original images   | - uploads/jobs/status        |
   | - store processed outputs | - metrics fields, timestamps  |
   +----------------------+    +------------------------------+

Observability (later, still OVH-hosted):
- Prometheus + Grafana (in cluster) + Alertmanager
- Logs: Loki (optional) or simple centralized logging
