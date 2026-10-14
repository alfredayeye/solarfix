# Cape Town Wind Factory — placeholder website

Static site presenting the Cape Town Vertical Wind Tunnel initiative to investors and sponsors:
the facility investment case, the Cape Town School of Human Flight, and the community transformation programme.

No build step, no framework. Edit the HTML and CSS directly and reload.

## Files

| Path | Purpose |
| --- | --- |
| `index.html` | Home: hero, three proposals, facility, market, timeline, leadership, FAQ, contact |
| `investment.html` | Investment & partnership proposal |
| `school-of-human-flight.html` | School of Human Flight sponsorship proposal |
| `community.html` | Community transformation / CSI proposal |
| `assets/css/site.css` | All styling. Colours and fonts are tokens at the top of the file |
| `assets/js/site.js` | Mobile menu toggle only |
| `assets/img/` | Optimised images (JPEG, ≤ 1800 px wide) |
| `sitemap.xml`, `robots.txt` | Search engine discovery |
| `llms.txt` | Plain-text summary for AI assistants and LLM crawlers |
| `favicon.svg` | Tab icon |
| `.github/workflows/pages.yml` | GitHub Pages deployment |

Each page carries a title, meta description, canonical URL, Open Graph and Twitter cards,
and JSON-LD structured data (Organization, WebSite, FAQPage on the home page, WebPage on the others).

## Before going live

1. **Domain.** The canonical URL, sitemap, robots.txt and llms.txt all use `https://capetownwindfactory.com`.
   Search and replace that string if a different domain is chosen.
2. **Contact email.** `invest@capetownwindfactory.com` in `index.html` is a placeholder. Replace it with a real mailbox.
3. **Hosting.** Any static host works (GitHub Pages, Cloudflare Pages, Netlify, Vercel).
   For GitHub Pages: Settings → Pages → Source: "GitHub Actions"; the included workflow deploys on every push to `main`.
4. **Imagery.** The facility rendering is a concept image. Replace with architect visuals when available.
5. **Figures.** The School of Human Flight investment table shows `TBC` placeholders, matching the proposal document.

## Previewing locally

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```
