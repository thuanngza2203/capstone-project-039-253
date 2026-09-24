import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const components = {
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
  table: ({ node, ...props }) => (
    <div className="table-scroll">
      <table {...props} />
    </div>
  ),
};

// Câu trả lời của LLM: in đậm, gạch đầu dòng, bảng. Không render HTML thô.
export default function Markdown({ children }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children || ""}
      </ReactMarkdown>
    </div>
  );
}
