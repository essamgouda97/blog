import { getPostBySlug, getAllPosts } from "@/lib/api"

export async function generateStaticParams() {
  const posts = await getAllPosts()
  return posts.map((post) => ({
    slug: post.slug,
  }))
}

export default async function Post({ params }: { params: { slug: string } }) {
  const post = await getPostBySlug(params.slug)

  return (
    <article className="container mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold mb-2">{post.title}</h1>
        <div className="text-sm text-muted-foreground">
          {post.date}
        </div>
      </header>
      <div className="prose prose-lg dark:prose-invert prose-pre:bg-secondary prose-pre:text-secondary-foreground max-w-none">
        <div 
          className="[&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
          dangerouslySetInnerHTML={{ __html: post.contentHtml || '' }} 
        />
      </div>
    </article>
  )
}

