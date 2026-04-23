import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { encodeShareUrl, decodeShareUrl } from '../utils/shareUrl';

/**
 * Property 12: Share URL round-trip
 *
 * For any non-empty string (including Hebrew characters), encoding into
 * a shareable URL and decoding back produces the original text unchanged.
 *
 * **Validates: Requirements 10.2**
 */
describe('Share URL round-trip', () => {
  it('should round-trip any non-empty string including Hebrew', () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 1, maxLength: 500 }),
        (text) => {
          const url = encodeShareUrl(text);
          const decoded = decodeShareUrl(url);
          expect(decoded).toBe(text);
        },
      ),
      { numRuns: 200 },
    );
  });

  it('should round-trip Hebrew-specific strings', () => {
    const hebrewChars = 'אבגדהוזחטיכלמנסעפצקרשת ';
    const hebrewArb = fc.array(
      fc.constantFrom(...hebrewChars.split('')),
      { minLength: 1, maxLength: 300 },
    ).map((chars) => chars.join(''));

    fc.assert(
      fc.property(hebrewArb, (text) => {
        const url = encodeShareUrl(text);
        const decoded = decodeShareUrl(url);
        expect(decoded).toBe(text);
      }),
      { numRuns: 200 },
    );
  });

  it('should round-trip mixed Hebrew and ASCII strings', () => {
    const hebrewArb = fc.array(
      fc.constantFrom(...'אבגדהוזחטיכלמנסעפצקרשת'.split('')),
      { minLength: 1, maxLength: 100 },
    ).map((chars) => chars.join(''));

    fc.assert(
      fc.property(
        fc.tuple(
          fc.string({ minLength: 1, maxLength: 100 }),
          hebrewArb,
        ),
        ([ascii, hebrew]) => {
          const text = hebrew + ' ' + ascii;
          const url = encodeShareUrl(text);
          const decoded = decodeShareUrl(url);
          expect(decoded).toBe(text);
        },
      ),
      { numRuns: 100 },
    );
  });

  it('should return null for URLs without text parameter', () => {
    expect(decodeShareUrl('https://example.com/')).toBeNull();
    expect(decodeShareUrl('https://example.com/?other=value')).toBeNull();
  });

  it('should return null for invalid base64 in text parameter', () => {
    expect(decodeShareUrl('https://example.com/?text=!!!invalid!!!')).toBeNull();
  });
});
