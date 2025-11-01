export function MiniAppResume() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-4xl font-black mb-6 border-b-[3px] border-black pb-2">Resume</h2>

      <div className="space-y-6">
        <div className="bg-white p-6 border-[3px] border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
          <h3 className="text-2xl font-black mb-4">Experience</h3>

          <div className="space-y-4">
            <div className="border-l-[4px] border-[#FF2E63] pl-4">
              <h4 className="text-xl font-bold">Senior Frontend Developer</h4>
              <p className="text-gray-600 font-medium">TechCorp Inc. • 2022 - Present</p>
              <p className="mt-2">
                Leading development of next-generation web applications using React, TypeScript, and modern tooling.
              </p>
            </div>

            <div className="border-l-[4px] border-[#FF2E63] pl-4">
              <h4 className="text-xl font-bold">Full Stack Developer</h4>
              <p className="text-gray-600 font-medium">StartupXYZ • 2020 - 2022</p>
              <p className="mt-2">
                Built scalable web applications from concept to deployment, working across the entire technology stack.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 border-[3px] border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
          <h3 className="text-2xl font-black mb-4">Projects</h3>

          <div className="grid gap-4">
            <div className="p-4 bg-gray-50 border-[2px] border-black">
              <h4 className="text-lg font-bold">AI-Powered Design Tool</h4>
              <p className="text-sm text-gray-600 mb-2">React, Python, OpenAI API</p>
              <p>Created an intelligent design assistant that helps users generate and iterate on creative concepts.</p>
            </div>

            <div className="p-4 bg-gray-50 border-[2px] border-black">
              <h4 className="text-lg font-bold">Real-time Collaboration Platform</h4>
              <p className="text-sm text-gray-600 mb-2">Next.js, WebSockets, PostgreSQL</p>
              <p>Built a platform enabling seamless real-time collaboration for distributed teams.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
