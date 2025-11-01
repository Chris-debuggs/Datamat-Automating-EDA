export function MiniAppAbout() {
  return (
    <div className="max-w-2xl">
      <h2 className="text-4xl font-black mb-6 border-b-[3px] border-black pb-2">About</h2>

      <div className="space-y-6">
        <div className="bg-white p-6 border-[3px] border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
          <h3 className="text-2xl font-black mb-4">Hey, I'm Manan! 👋</h3>
          <p className="text-lg leading-relaxed mb-4">
            I'm a creative developer who loves building things that matter. I work at the intersection of design and
            technology, creating experiences that are both beautiful and functional.
          </p>
          <p className="text-lg leading-relaxed">
            When I'm not coding, you'll find me exploring new art forms, writing about technology, or experimenting with
            AI and creative tools.
          </p>
        </div>

        <div className="bg-white p-6 border-[3px] border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
          <h3 className="text-2xl font-black mb-4">Skills & Interests</h3>
          <div className="flex flex-wrap gap-2">
            {["React", "Next.js", "TypeScript", "AI/ML", "Design Systems", "Creative Coding", "Writing", "Art"].map(
              (skill) => (
                <span
                  key={skill}
                  className="bg-[#FF2E63] text-white px-3 py-1 border-[2px] border-black font-bold text-sm"
                >
                  {skill}
                </span>
              ),
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
