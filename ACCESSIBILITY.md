# Nethical Accessibility Statement (WCAG 2.2 / EAA Compliance)

**Effective Date:** 2026-09-15  
**Version:** 2.0  
**Standards:** Web Content Accessibility Guidelines (WCAG) 2.2 Level AA, European Accessibility Act (Directive (EU) 2019/882), US Section 508 of the Rehabilitation Act  

---

## 1. Our Commitment

Nethical is dedicated to ensuring digital accessibility for people of all abilities. We believe that sovereign AI governance and safety tools must be accessible to every engineer, compliance auditor, and node operator.

We continually improve the user experience for everyone and apply relevant accessibility standards across the **Nethical Web Portal (Błyskawica Sovereign Control Plane)**, documentation portals, and CLI interfaces.

---

## 2. Conformance Status

The Nethical Web Portal is designed to conform with **WCAG 2.2 Level AA** standards.

### Key Accessibility Features Implemented:
1. **High Contrast & Visual Hierarchy:**
   - Default theme exceeds the 4.5:1 contrast ratio requirement for body text and 3:1 for interactive UI components and graphical objects.
   - High-contrast mode toggle available directly in navigation.
2. **Keyboard Navigability:**
   - All interactive controls, governance switches, and modal dialogs are fully navigable via `Tab`, `Shift+Tab`, `Enter`, and `Spacebar`.
   - Visible keyboard focus indicators with high-luminance outlines (`--cyan-accent`).
3. **Screen Reader Compatibility:**
   - Semantic HTML5 structure with ARIA landmarks (`role="main"`, `role="navigation"`, `role="alert"`).
   - Live region updates (`aria-live="polite"`) for real-time safety violation alerts and ledger stream changes.
4. **Motion & Reduced Motion:**
   - Respects user operating system preference (`@media (prefers-reduced-motion: reduce)`), disabling ambient particle animations and pulsating effects.
5. **Accessible CLI Tooling:**
   - CLI outputs support plain-text, monochromatic modes (`--no-color`) for screen readers and high-contrast terminal emulators.

---

## 3. Known Limitations & Ongoing Improvements

While we strive for comprehensive Level AA adherence:
- Some complex multi-dimensional Merkle-DAG network graph visualizations require supplementary tabular representations. An accessible tabular data view is provided alongside each visual graph.

---

## 4. Feedback & Contact Information

We welcome feedback on the accessibility of Nethical. If you encounter accessibility barriers, please let us know:
- **Email:** `accessibility@nethical.ai`
- **GitHub Issues:** Open an issue with the `accessibility` label.
- **Response SLA:** We aim to acknowledge and address accessibility concerns within five (5) business days.
