/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
    typography: {
      DEFAULT: {
        css: {
          color: 'hsl(var(--foreground))',
          h1: {
            color: 'hsl(var(--foreground))',
            fontWeight: '700',
            fontSize: '2.25em',
            marginTop: '0',
            marginBottom: '0.8888889em',
            lineHeight: '1.1111111'
          },
          h2: {
            color: 'hsl(var(--foreground))',
            fontWeight: '600',
            fontSize: '1.5em',
            marginTop: '2em',
            marginBottom: '1em',
            lineHeight: '1.3333333'
          },
          'blockquote p:first-of-type::before': { content: 'none' },
          'blockquote p:last-of-type::after': { content: 'none' },
          blockquote: {
            borderLeftColor: 'hsl(var(--primary))',
            borderLeftWidth: '2px',
            fontWeight: '400',
            fontStyle: 'normal',
            quotes: 'none',
            marginTop: '1.6em',
            marginBottom: '1.6em',
            paddingLeft: '1em'
          },
          code: {
            backgroundColor: 'hsl(var(--secondary))',
            padding: '0.25rem',
            borderRadius: '0.25rem',
            fontSize: '0.875em',
            fontWeight: '400',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
          },
          'code::before': { content: '""' },
          'code::after': { content: '""' },
          pre: {
            backgroundColor: 'hsl(var(--secondary))',
            color: 'hsl(var(--secondary-foreground))',
            fontSize: '0.875em',
            lineHeight: '1.7142857',
            marginTop: '1.7142857em',
            marginBottom: '1.7142857em',
            borderRadius: '0.375rem',
            paddingTop: '0.8571429em',
            paddingRight: '1.1428571em',
            paddingBottom: '0.8571429em',
            paddingLeft: '1.1428571em'
          },
          'pre code': {
            backgroundColor: 'transparent',
            borderWidth: '0',
            borderRadius: '0',
            padding: '0',
            fontWeight: '400',
            color: 'inherit',
            fontSize: 'inherit',
            fontFamily: 'inherit',
            lineHeight: 'inherit'
          },
          strong: {
            color: 'hsl(var(--foreground))',
            fontWeight: '600'
          },
          a: {
            color: 'hsl(var(--primary))',
            textDecoration: 'underline',
            fontWeight: '500'
          },
          'ul > li': {
            paddingLeft: '1.5em',
            position: 'relative'
          },
          'ul > li::before': {
            content: '""',
            position: 'absolute',
            backgroundColor: 'hsl(var(--foreground))',
            borderRadius: '50%',
            width: '0.375em',
            height: '0.375em',
            top: 'calc(0.875em - 0.1875em)',
            left: '0.25em'
          },
          img: {
            borderRadius: '0.375rem'
          },
          hr: {
            borderColor: 'hsl(var(--border))',
            marginTop: '3em',
            marginBottom: '3em'
          }
        }
      }
    }
  },
  plugins: [
    require("tailwindcss-animate"),
    require("@tailwindcss/typography"),
  ],
}

