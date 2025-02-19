import fs from "fs"
import path from "path"
import matter from "gray-matter"
import { remark } from "remark"
import html from "remark-html"

const contentDirectory = path.join(process.cwd(), "content")

export type Post = {
  slug: string
  title: string
  date: string
  excerpt: string
  content: string
  contentHtml?: string
  coverImage?: string
}

export async function getPostBySlug(slug: string): Promise<Post> {
  const fullPath = path.join(contentDirectory, slug, "index.md")
  const fileContents = fs.readFileSync(fullPath, "utf8")
  const { data, content } = matter(fileContents)
  
  // Convert markdown to HTML with custom image handling
  const processedContent = await remark()
    .use(() => (tree) => {
      // Find all images in the markdown
      const images = tree.children.filter(node => 
        node.type === 'paragraph' && 
        node.children?.[0]?.type === 'image'
      )
      
      // Remove standalone images from their original position
      tree.children = tree.children.filter(node => 
        !(node.type === 'paragraph' && 
          node.children?.length === 1 && 
          node.children[0].type === 'image')
      )
      
      // Add images back at the end of the content
      tree.children.push(...images)
    })
    .use(html)
    .process(content)
    
  const contentHtml = processedContent.toString()
  
  // Check if there's a cover image in the static folder
  const staticDir = path.join(contentDirectory, slug, "static")
  let coverImage: string | undefined
  
  if (fs.existsSync(staticDir)) {
    const files = fs.readdirSync(staticDir)
    const coverFile = files.find(file => 
      file.toLowerCase().includes("cover") || 
      files[0]
    )
    if (coverFile) {
      coverImage = `/content/${slug}/static/${coverFile}`
    }
  }

  return {
    slug,
    title: data.title,
    date: data.date,
    excerpt: data.excerpt || "",
    content,
    contentHtml,
    coverImage,
  }
}

export async function getAllPosts(): Promise<Post[]> {
  const slugs = fs.readdirSync(contentDirectory)
    .filter(dir => 
      fs.statSync(path.join(contentDirectory, dir)).isDirectory() &&
      fs.existsSync(path.join(contentDirectory, dir, "index.md"))
    )

  const posts = await Promise.all(
    slugs.map((slug) => getPostBySlug(slug))
  )

  return posts.sort((post1, post2) => (post1.date > post2.date ? -1 : 1))
}

