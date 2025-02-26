import fs from "fs"
import path from "path"
import matter from "gray-matter"

const contentDirectory = path.join(process.cwd(), "content")

export type Post = {
  slug: string
  title: string
  date: string
  excerpt: string
  content: string  // Changed back to string
  coverImage?: string
}

export async function getPostBySlug(slug: string): Promise<Post> {
  const fullPath = path.join(contentDirectory, slug, "index.md")
  const fileContents = fs.readFileSync(fullPath, "utf8")
  const { data, content } = matter(fileContents)

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
    content,  // Return raw content
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

