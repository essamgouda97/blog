import { getAllPosts } from "@/lib/api"
import Link from "next/link"

export default async function Home() {
  const posts = await getAllPosts()

  return (
    <div>
      <h1 className="text-4xl font-bold mb-8">Latest Posts</h1>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {posts.map((post) => (
          <Link key={post.slug} href={`/blog/${post.slug}`} className="block">
            <div className="border border-primary rounded-lg p-6 hover:bg-primary/10 transition-colors min-h-[200px] flex flex-col">
              <h2 className="text-2xl font-semibold mb-2">{post.title}</h2>
              <p className="text-muted-foreground flex-grow">{post.excerpt}</p>
              <p className="text-sm text-muted-foreground mt-4">{post.date}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

