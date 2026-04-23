/**
 * Share URL utilities — encode/decode Hebrew text into a shareable URL.
 *
 * Uses base64 encoding of the UTF-8 text bytes as a query parameter.
 * This supports the full Unicode range including Hebrew characters.
 */

/**
 * Encode text into a shareable URL with the text as a base64 query param.
 *
 * @param text - The text to encode (any string including Hebrew)
 * @returns A full URL string with `?text=<base64>` query parameter
 */
export function encodeShareUrl(text: string): string {
  // Encode the text as UTF-8 bytes, then base64-encode
  const encoded = btoa(
    encodeURIComponent(text).replace(/%([0-9A-F]{2})/g, (_match, p1) =>
      String.fromCharCode(parseInt(p1, 16)),
    ),
  );
  const base = typeof window !== 'undefined' ? window.location.origin : 'https://example.com';
  return `${base}/?text=${encodeURIComponent(encoded)}`;
}

/**
 * Decode text from a shareable URL.
 *
 * @param url - The full URL or just the query string containing `text=<base64>`
 * @returns The decoded text string, or null if the URL doesn't contain valid encoded text
 */
export function decodeShareUrl(url: string): string | null {
  try {
    const urlObj = new URL(url, 'https://example.com');
    const encoded = urlObj.searchParams.get('text');
    if (!encoded) return null;

    const decoded = decodeURIComponent(encoded);
    // Decode base64 back to UTF-8 string
    const bytes = atob(decoded);
    const text = decodeURIComponent(
      bytes
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join(''),
    );
    return text || null;
  } catch {
    return null;
  }
}
