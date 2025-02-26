/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/content/:path*',
        destination: '/api/content/:path*',
      },
    ]
  },
}

export default nextConfig
