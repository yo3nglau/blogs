# Academic Personal Website — Design Spec

**Date:** 2026-03-30
**Status:** Approved

---

## Overview

Create a new standalone Hugo academic personal website in a separate GitHub repository (`yo3nglau/academic`), without any blog functionality. The site reuses the XMin theme and visual style of the existing blog site.

**Live URL:** `https://yo3nglau.github.io/academic/`

---

## Architecture

- **Local path:** `/mnt/e/GitRepository/academic/`
- **GitHub repo:** `yo3nglau/academic`
- **Build tool:** Hugo with XMin theme (git submodule)
- **Publish directory:** `docs/` (read by GitHub Pages)
- **`baseURL`:** `/academic/`
- **Site title:** `Yang Liu`

---

## Pages

| Page | File | Nav Label |
|------|------|-----------|
| Home | `content/_index.md` | Home |
| Publications | `content/publications.md` | Publications |
| Awards | `content/awards.md` | Awards |
| Activities | `content/activities.md` | Activities |

No Categories, Tags, or post listings.

---

## Content

### Home (`content/_index.md`)
Migrated from current `about.md`:
- **Interest:** Artificial Intelligence for Healthcare
- **Education:** Tsinghua University (Ph.D., 2022–Present), Shandong University (B.E., 2018–2022)
- **Contact:** lyang22ATmails.tsinghua.edu.cn

### Publications (`content/publications.md`)
Migrated from current `about.md`:
- [1] Liu, Y. et al. (2025). NPJ Digital Medicine. DOI link.
- [2] Liu, Y. et al. (2023). Computers in Biology and Medicine. DOI link.
- [3] Liu, Y. et al. (2022). Frontiers in Computer Science. DOI link.
- Patent: CN202310272410.0 (2023).

### Awards (`content/awards.md`)
Migrated from current `about.md`:
- National Scholarship of China, 2024
- Outstanding Graduate of Shandong Province, 2022
- National Scholarship of China, 2021
- National Scholarship of China, 2020

### Activities (`content/activities.md`)
Migrated from current `about.md`:
- Academic visit at Cambridge, Oxford, Imperial College London, Cardiff University, 2025.10
- Speech at iBHE welcome meeting, SIGS, Tsinghua University, 2025.9
- Academic visit at HKUST, 2025.3

---

## Styling

- Copy `static/css/style.css` and `static/css/fonts.css` from the existing site unchanged.
- Copy `layouts/partials/header.html`, `footer.html`, and `layouts/_default/single.html` from the existing site.
- Navigation menu contains exactly 4 items: Home, Publications, Awards, Activities.
- No PDF.js or other extra static assets.

---

## Configuration (`config.yaml`)

```yaml
baseurl: "/academic/"
languageCode: "en-us"
title: "Yang Liu"
publishDir: "docs"

menu:
  main:
    - name: Home
      url: ""
      weight: 1
    - name: Publications
      url: "publications/"
      weight: 2
    - name: Awards
      url: "awards/"
      weight: 3
    - name: Activities
      url: "activities/"
      weight: 4

params:
  footer: "&copy; Yang Liu {Year}"

markup:
  goldmark:
    renderer:
      unsafe: true
```

---

## Out of Scope

- Blog posts, categories, tags
- PDF embedding
- Search functionality
- Dark mode
