import { getPostBySlug, getAllPosts } from "@/lib/api"
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'

export async function generateStaticParams() {
  const posts = await getAllPosts()
  return posts.map((post) => ({
    slug: post.slug,
  }))
}

export default async function Post({ params }: { params: { slug: string } }) {
  const post = await getPostBySlug(params.slug)

  return (
    <article className="max-w-4xl mx-auto">
      <header className="mb-12">
        <h1 className="text-4xl font-bold mb-4">{post.title}</h1>
        <div className="text-sm text-muted-foreground">
          {post.date}
        </div>
      </header>
      <div className="prose prose-lg dark:prose-invert prose-pre:bg-secondary prose-pre:text-secondary-foreground max-w-none">
        <ReactMarkdown 
          remarkPlugins={[remarkMath, remarkGfm]}
          rehypePlugins={[rehypeKatex]}
          components={{
            h2: ({node, ...props}) => <h2 className="mt-12 mb-6" {...props} />,
            h3: ({node, ...props}) => <h3 className="mt-8 mb-4" {...props} />,
            p: ({node, ...props}) => <p className="mb-6 leading-relaxed" {...props} />,
            img: ({node, ...props}) => (
              <div className="my-12">
                <img
                  className="rounded-lg mx-auto max-h-[600px] w-auto"
                  {...props}
                  loading="lazy"
                />
              </div>
            ),
            ul: ({node, ...props}) => <ul className="my-6 space-y-2" {...props} />,
            ol: ({node, ...props}) => <ol className="my-6 space-y-2" {...props} />,
            pre: ({node, ...props}) => <pre className="my-8 p-4 rounded-lg" {...props} />,
          }}
        >
          {post.content}
        </ReactMarkdown>
      </div>
    </article>
  )
}

