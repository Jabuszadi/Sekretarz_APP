    // frontend/tailwind.config.cjs
    /** @type {import('tailwindcss').Config} */
    module.exports = {
      content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}", // To jest kluczowa linia!
      ],
      theme: {
        extend: {},
      },
      plugins: [],
    }